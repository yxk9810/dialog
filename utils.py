import re
def cut_sentences(line):
    def clean(txt):
        txt = txt.lower()
        txt = re.sub('(\s*)?(<.*?>)?', '', txt)
        return txt

    ordinary_sent_seg_symbol = list('。!！？；;?')
    special_sent_seg_symbol = ['＜ｂｒ＞', '。”', '！”']
    sent_seg_symbol = set(ordinary_sent_seg_symbol + special_sent_seg_symbol)
    seg_re_pattern = u'([{}])?'.format(''.join(ordinary_sent_seg_symbol)) + u''.join(
        [u'({})?'.format(x) for x in special_sent_seg_symbol])

    sents = re.split(seg_re_pattern, line, flags=re.U)
    sents = [x for x in sents if x is not None and len(x) > 0]
    if len(sents) == 0:
        return []
    output = [sents[0]]
    for sent in sents[1:]:
        if sent in sent_seg_symbol or len(list(clean(sent))) < 1:
            output[-1] += sent
        else:
            output.append(sent)
    if len(output) > 1:
        if len(list(clean(output[-1]))) < 3 or output[-1] in sent_seg_symbol:
            output[-2] = output[-2] + output[-1]
            output = output[0:-1]
    sentences = []
    sent = ''
    for tok in output:
        if tok[-1] in sent_seg_symbol or tok[-1].strip() in ordinary_sent_seg_symbol:
            sent += tok
            sentences.append(sent)
            sent = ''
        else:
            sent += tok
    if sent != '': sentences.append(sent)
    return sentences