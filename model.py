import torch
import torch.nn as nn


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size):
        super(PointerNet, self).__init__()

        self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

    def forward(self, src_encodings, src_token_mask, query_vec,
                log=False, return_logits=False):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(batch_size, tgt_action_num, query_vec_size)
        :return: Variable(batch_size, tgt_action_num, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)
        src_trans = self.src_encoding_linear(src_encodings).unsqueeze(1)
        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.unsqueeze(3)

        # (batch_size, tgt_action_num, src_sent_len)
        weights = torch.matmul(src_trans, q).squeeze(3)

        if src_token_mask is not None:
            src_token_mask = (1-src_token_mask).unsqueeze(1).expand_as(weights)
            weights = weights.masked_fill(src_token_mask.bool(), -1e4)

        if return_logits:
            return weights

        if log:
            ptr_weights = torch.log_softmax(weights, dim=-1)
        else:
            ptr_weights = torch.softmax(weights, dim=-1)

        return ptr_weights


class RNN_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(config.vocab_size, config.hid_size, padding_idx=config.pad_id)
        dropout = config.dropout if config.layers > 1 else 0.0
        self.gru = nn.GRU(config.hid_size, config.hid_size, config.layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(config.hid_size * 2, config.hid_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input, mask=None):
        """
        input: [B, LL, SL]
        mask: [B, LL]
        """
        batch_size, l_size, s_size = input.size()
        t_input = self.embedder(input.view(batch_size * l_size, s_size))
        _, h = self.gru(t_input)    # h = [2, B*LL, hid_size]
        # RNN输入序列短的情况下根据leng不一定提升效果,因为梯度爆炸和梯度消失不严重
        # transpose或者permute后必须用contiguous + view或直接reshape
        out = h.permute(1, 0, 2).reshape([batch_size, l_size, -1])
        # 一般dropout放在激活函数之后
        out = self.fc(self.dropout(self.tanh(out)))
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


class AstDiff2Seq(nn.Module):
    def __init__(self,
                 config,
                 embedder=None,
                 node_embedder=None,
                 encoder=None,
                 change_encoder=None,
                 decoder=None,
                 pointer_net=None,
                 beam_size=None,
                 max_length=None):
        super(AstDiff2Seq, self).__init__()
        if embedder is None:
            embedder = RNN_Embedding(config.embedder_config)
        if node_embedder is None:
            node_embedder = RNN_Embedding(config.node_embedder_config)
        if encoder is None:
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.hid_size, nhead=config.attn_heads,
                                                       dim_feedforward=config.intermediate_size)
            encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.enc_layers)
        if change_encoder is None:
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.hid_size, nhead=config.attn_heads,
                                                       dim_feedforward=config.intermediate_size)
            change_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.chg_layers)
        if decoder is None:
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.hid_size, nhead=config.attn_heads,
                                                       dim_feedforward=config.intermediate_size)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.dec_layers)
        if pointer_net is None:
            pointer_net = PointerNet(config.hid_size, config.hid_size)
        self.embedder = embedder
        self.node_embedder = node_embedder
        self.pos_embedder = nn.Embedding(1000, config.hid_size)
        self.register_buffer('position_ids', torch.arange(1000).expand((1, -1)))
        # tgt_embedding时光考虑vocab_size是不够的，还要把copy_id包括进去
        self.tgt_embedder = nn.Embedding(config.vocab_size+(config.max_pos*2), config.hid_size)
        self.diff_embedder = nn.Embedding(config.diff_vocab_size, config.hid_size)
        self.encoder = encoder
        self.change_encoder = change_encoder
        self.decoder = decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hid_size, nhead=config.attn_heads,
                                                    dim_feedforward=config.intermediate_size)
        self.diff_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.enc_layers)
        self.register_buffer("bias", torch.tril(torch.ones(1000, 1000)))
        self.pointer_net = pointer_net
        self.config = config
        self.tanh = nn.Tanh()
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.dense = nn.Linear(config.hid_size, config.hid_size)
        self.chg_embedder = nn.Embedding(config.chg_vocab, config.hid_size)
        self.layernorm = nn.LayerNorm(config.hid_size, eps=config.layer_norm_eps)
        self.layernorm1 = nn.LayerNorm(config.hid_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hid_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.dense1 = nn.Linear(config.hid_size, config.hid_size)
        # 最后预测时留两个max_pos作为拷贝位置
        self.tgt_size = config.vocab_size + config.max_pos * 2
        self.register_buffer('gen_mask', torch.tensor([[0.]*config.vocab_size+[-1e4]*(config.max_pos*2)]))
        self.lm_head = nn.Linear(config.hid_size, self.tgt_size)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.gen_copy_switch = nn.Linear(config.hid_size, 3)
        self.register_buffer('zero', torch.LongTensor(1).fill_(0))

    # 这里输入的mask是有效位为1
    def forward(self,
                diff_ids=None,
                sub_token_ids1=None,
                ast_att_ids1=None,
                mask1=None,
                sub_token_ids2=None,
                ast_att_ids2=None,
                mask2=None,
                chg_idxs1=None,
                chg_mask1=None,
                chg_idxs2=None,
                chg_mask2=None,
                chg_type=None,
                copy_ids1=None,
                copy_ids2=None,
                tgt_ids=None,
                tgt_mask=None,
                ):
        # 处理diff
        diff_mask = diff_ids.eq(0) # pad_id=0,故pad位为Ture
        diff_embed = self.diff_embedder(diff_ids) + self.pos_embedder(self.position_ids[:, :diff_ids.size(1)])
        diff_embed = self.dropout(self.layernorm(diff_embed)).permute([1,0,2]).contiguous()
        diff = self.diff_encoder(src=diff_embed, src_key_padding_mask=diff_mask)  # batch_first=False
        # 为版本1的代码建立embedding
        b_s, l1, _ = sub_token_ids1.size()
        code_embed1 = self.embedder(sub_token_ids1)
        ast_embed1 = self.node_embedder(ast_att_ids1)
        pos_embed = self.pos_embedder(self.position_ids[:, :l1])
        embed1 = self.tanh(self.dropout(code_embed1 + ast_embed1 + pos_embed))
        embed1 = embed1 * mask1.unsqueeze(-1).float()
        # 为版本2的代码建立embedding，默认sub_token_ids1和sub_tokens_ids2的shape一样
        code_embed2 = self.embedder(sub_token_ids2)
        ast_embed2 = self.node_embedder(ast_att_ids2)
        embed2 = self.tanh(self.dropout(code_embed2 + ast_embed2 + pos_embed))
        embed2 = embed2 * mask2.unsqueeze(-1).float()
        # 将两个版本代码分别输入同一个encoder
        out1 = self.encoder(src=embed1.permute([1, 0, 2]).contiguous(), src_key_padding_mask=mask1.eq(0))
        out2 = self.encoder(src=embed2.permute([1, 0, 2]).contiguous(), src_key_padding_mask=mask2.eq(0))
        # 将out转为batch first形式
        bf_out1 = out1.permute([1, 0, 2]).contiguous()
        bf_out2 = out2.permute([1, 0, 2]).contiguous()
        # 使用gather根据chg_idxs获取变更的代码
        chg1 = bf_out1.gather(1, chg_idxs1.unsqueeze(2).expand(-1, -1, self.config.hid_size))
        chg2 = bf_out2.gather(1, chg_idxs2.unsqueeze(2).expand(-1, -1, self.config.hid_size))
        chg1 = chg1 * chg_mask1.unsqueeze(-1).float()
        chg2 = chg2 * chg_mask2.unsqueeze(-1).float()
        # 合并change的表示(加上chg_type)
        type_embed = self.chg_embedder(chg_type)
        chg_seq = self.dense(self.dropout(self.tanh((chg1 + chg2 + type_embed))))
        chg_seq = self.layernorm1(chg_seq).permute([1, 0, 2]).contiguous()
        chg_mask = chg_type.eq(0)  # chg_type中的0是pad补充的
        # 再次用Transformer Encoder编码
        memory = self.change_encoder(src=chg_seq, src_key_padding_mask=chg_mask)
        # 将代码和change进行拼接
        memory = torch.cat([diff, memory, out1, out2], dim=0)
        mask = torch.cat([diff_mask, chg_mask, ~mask1, ~mask2], dim=-1)

        if tgt_ids is not None:
            t_l = tgt_ids.size(1)
            attn_mask = -1e4 * (1-self.bias[:t_l, :t_l])

            tgt_embed = self.tgt_embedder(tgt_ids) + self.pos_embedder(self.position_ids[:, :t_l])
            tgt_embed = self.dropout(self.layernorm2(tgt_embed)).permute([1, 0, 2]).contiguous()

            out = self.decoder(tgt_embed, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask)
            # [batch size, tgt len, hid size]
            out = out.permute([1, 0, 2]).contiguous()

            # [batch size, tgt len, (gen, copy from seq1, copy from seq2)]
            token_gen_copy_switch = self.lsm(self.gen_copy_switch(out))
            gen, copy1, copy2 = torch.chunk(token_gen_copy_switch, chunks=3, dim=-1)

            # [batch size, tgt len, src len] without log only softmax
            copy_prob1 = self.pointer_net(src_encodings=bf_out1, src_token_mask=mask1.int(), query_vec=out)
            copy_prob2 = self.pointer_net(src_encodings=bf_out2, src_token_mask=mask2.int(), query_vec=out)

            hidden_states = torch.tanh(self.dense1(out))
            # [batch size, tgt len, vocab size] 直接进行生成的概率
            gen_logprob = self.lsm(self.lm_head(hidden_states)) + self.gen_mask.unsqueeze(1) + gen

            copy_ids1_ = copy_ids1.unsqueeze(1).expand_as(copy_prob1)
            copy_ids2_ = copy_ids2.unsqueeze(1).expand_as(copy_prob1)
            copy_logprob1 = torch.full_like(gen_logprob, 1e-6).scatter_add_(2, copy_ids1_, copy_prob1).log()
            copy_logprob2 = torch.full_like(gen_logprob, 1e-6).scatter_add_(2, copy_ids2_, copy_prob2).log()

            logprob = torch.stack([gen_logprob, copy_logprob1+copy1, copy_logprob2+copy2], dim=-1)
            logprob = torch.logsumexp(logprob, dim=-1)

            # Shift so that tokens < n predict n
            active_loss = tgt_mask[..., 1:].ne(0).view(-1) == 1
            shift_logprob = logprob[..., :-1, :].contiguous()
            shift_labels = tgt_ids[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.NLLLoss(ignore_index=-1)
            # 这里的[active_loss]表示只保存True所在位置的值
            loss = loss_fct(shift_logprob.view(-1, shift_logprob.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            outputs = loss, loss*active_loss.sum(), active_loss.sum()
            return outputs
        else:
            preds = []
            for i in range(b_s):
                context = memory[:, i:i+1]
                context_mask = mask[i:i+1]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                src1 = bf_out1[i:i+1].repeat(self.beam_size, 1, 1)
                src2 = bf_out2[i:i+1].repeat(self.beam_size, 1, 1)
                src_mask1 = mask1[i:i+1].repeat(self.beam_size, 1)
                src_mask2 = mask2[i:i+1].repeat(self.beam_size, 1)
                copy_ids1_ = copy_ids1[i:i+1].repeat(self.beam_size, 1)  # [beam size, src len]
                copy_ids2_ = copy_ids2[i:i+1].repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    t_l = input_ids.size(1)
                    attn_mask = -1e4 * (1-self.bias[:t_l, :t_l])
                    tgt_embed = self.tgt_embedder(input_ids) + self.pos_embedder(self.position_ids[:, :t_l])
                    tgt_embed = self.dropout(self.layernorm2(tgt_embed)).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embed, context, tgt_mask=attn_mask, memory_key_padding_mask=context_mask)
                    out = out.permute([1, 0, 2]).contiguous()
                    token_gen_copy_switch = self.lsm(self.gen_copy_switch(out))[:, -1, :]  # [beam size, 3]
                    gen, copy1, copy2 = torch.chunk(token_gen_copy_switch, chunks=3, dim=-1)
                    copy_prob1 = self.pointer_net(src_encodings=src1, src_token_mask=src_mask1.int(), query_vec=out)
                    copy_prob2 = self.pointer_net(src_encodings=src2, src_token_mask=src_mask2.int(), query_vec=out)

                    hidden_states = torch.tanh(self.dense1(out))[:, -1, :]
                    gen_logprob = self.lsm(self.lm_head(hidden_states)) + self.gen_mask + gen  # [beam size, vocabs size]

                    copy_logprob1 = torch.full_like(gen_logprob, 1e-6).scatter_add_(1, copy_ids1_, copy_prob1[:, -1, :]).log()
                    copy_logprob2 = torch.full_like(gen_logprob, 1e-6).scatter_add_(1, copy_ids2_, copy_prob2[:, -1, :]).log()

                    logprob = torch.stack([gen_logprob, copy_logprob1+copy1, copy_logprob2+copy2], dim=-1)
                    logprob = torch.logsumexp(logprob, dim=-1).data

                    beam.advance(logprob)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p]+[self.zero]*(self.max_length-len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


def test_RNN_Embedding():
    from utils import dotdict
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = dotdict()
    config.vocab_size = 100
    config.hid_size = 5
    config.layers = 1
    config.dropout = 0.4
    config.pad_id = 0
    embedder = RNN_Embedding(config)
    a = torch.ones(2, 3, 4).long()
    mask = torch.zeros(2, 3).long()
    print(embedder(a, mask))


def test_AstDiff2Seq():
    from utils import dotdict
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config1 = dotdict(vocab_size=100,
                      hid_size=64,
                      pad_id=2,
                      layers=1,
                      dropout=0.4,)
    config2 = dotdict(vocab_size=50,
                      hid_size=64,
                      pad_id=2,
                      layers=1,
                      dropout=0.4,)
    config3 = dotdict(embedder_config=config1,
                      node_embedder_config=config2,
                      hid_size=64,
                      vocab_size=300,
                      diff_vocab_size=200,
                      attn_heads=8,
                      intermediate_size=256,
                      enc_layers=3,
                      chg_layers=1,
                      dec_layers=3,
                      max_pos=250,
                      layer_norm_eps=1e-6,
                      sos_id=1,
                      eos_id=2,
                      pad_id=0,
                      chg_vocab=6,
                      dropout=0.1,)
    device = torch.device('cuda')
    astdiff2seq = AstDiff2Seq(config=config3, beam_size=5, max_length=10).to(device)
    diff_ids = torch.zeros(2, 10).long().to(device)
    diff_ids[0,0]=1
    sub_token_ids = torch.ones(2, 32, 4).long().to(device)
    mask = torch.ones(2, 32).bool().to(device)
    chg_idxs = torch.ones(2, 8).long().to(device)
    copy_ids = torch.ones(2, 32).long().to(device)
    tgt_ids = torch.ones(2, 10).long().to(device)
    print(astdiff2seq(
        diff_ids=diff_ids,
        sub_token_ids1=sub_token_ids,
        ast_att_ids1=sub_token_ids,
        mask1=mask,
        sub_token_ids2=sub_token_ids,
        ast_att_ids2=sub_token_ids,
        mask2=mask,
        chg_idxs1=chg_idxs,
        chg_mask1=chg_idxs,
        chg_idxs2=chg_idxs,
        chg_mask2=chg_idxs,
        chg_type=chg_idxs,
        copy_ids1=copy_ids,
        copy_ids2=copy_ids,
        tgt_ids=tgt_ids,
        tgt_mask=tgt_ids.bool().to(device),
    ))


if __name__ == '__main__':
    # test_RNN_Embedding()
    test_AstDiff2Seq()
