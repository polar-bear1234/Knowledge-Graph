#%%
import torch
import pickle
from utils import load_obj, tensorized


def predict(model, text):
    text_list = list(text)
    text_list.append("<end>")
    text_list = [text_list]
    crf_word2id = load_obj('crf_word2id')
    crf_tag2id = load_obj('crf_tag2id')
    # vocab_size = len(crf_word2id)
    # out_size = len(crf_tag2id)
    pred_tag_lists = model.predict(text_list, crf_word2id, crf_tag2id)
    return pred_tag_lists[0]


def result_process(text_list, tag_list):
    tuple_result = zip(text_list, tag_list)
    sent_out = []
    tags_out = []
    outputs = []
    words = ""
    for s, t in tuple_result:
        if t.startswith('B-') or t == 'O':
            if len(words):
                sent_out.append(words)
                # print(sent_out)
            if t != 'O':
                tags_out.append(t.split('-')[1])
            else:
                tags_out.append(t)
            words = s
            # print(words)
        else:
            words += s
    if len(sent_out) < len(tags_out):
        sent_out.append(words)
    outputs.append(''.join([str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs, [*zip(sent_out, tags_out)]


#%%
if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    modelpath = './ckpts/bilstm_crf.pkl'
    f = open(modelpath, 'rb')
    s = f.read()
    model = pickle.loads(s)

    # 测试样例
    # text = "INST_ID为1.00的活动会话数较高，为:27.00个"
    # text = "[OS Windows] 10.7.33.203 服务器已经超过90天未重启。请关注。"
    # text = "【间隔压缩告警】 appsystem:yzt,hostname:yzt-dgv-app-1，内容：SQLException告警20220830 14:08:49 [TID:43593.1574843.16618397294676427] ERROR [com.alibaba.druid.pool.PreparedStatementPool] - #[LOG:] exitImplicitCacheToClose error#java.sql.SQLException: 关闭的语句##011at oracle.jdbc.driver.OracleClosedStatement.exitImplicitCacheToClose(OracleClosedStatement.java:2967)##011at oracle.jdbc.driver.OraclePreparedStatementWrapper.exitImplicitCacheToClose(OraclePreparedStatementWrapper.java:1259)##011at com.alibaba.druid.util.OracleUtils.exitImplicitCacheToClose(OracleUtils.java:87)##011at com.alibaba.druid.pool.PreparedStatementPool.closeRemovedStatement(PreparedStatementPool.java:170)##011at com.alibaba.druid.pool.PreparedStatementPool.clear(PreparedStatementPool.java:138)##011at com.alibaba.druid.pool.DruidConnectionHolder.clearStatementCache(DruidConnectionHolder.java:255)##011at com.alibaba.druid.pool.DruidPooledConnection.disable(DruidPooledConnection.java:220)##011at com"
    # text = "[OS Linux] 10.10.14.11 CPU iowait时间占用的比率 5分钟平均值大于等于20%"
    text = "聚安一站通mem_meminfo_total在时间2023-02-08 08:00:05发生失败，当前值大于小于20%,请关注!"   #

    tag_res = predict(model, text)
    result, tuple_re = result_process(list(text), tag_res)
    # print(tuple_re)
    print(text)
    result = []
    tag = []
    for s, t in tuple_re:
        if t != 'O':
            result.append(s)
            tag.append(t)
    print([*zip(result, tag)])
    # print(dict(zip(tag, result)))

"""
    # others
    # ----------------------------------
    # s_path = "/Users/zyl/Desktop/competition/中文命名实体识别1/My_LSTM-CRF/data/tb_alert1115_1120.csv"
    s_path = "/Users/zyl/Desktop/competition/中文命名实体识别1/My_LSTM-CRF/data/tb_alert0207_0209.csv"
    df = pd.read_csv(s_path)
    Texts = df.content.values.tolist()
    times = df.event_time.values.tolist()
    # save_ = "/Users/zyl/Desktop/competition/中文命名实体识别1/My_LSTM-CRF/data/15_20.csv"
    save_ = "/Users/zyl/Desktop/competition/中文命名实体识别1/My_LSTM-CRF/data/7_9.csv"
    fout = []
    for i, text in tqdm(enumerate(Texts)):
        tag_res = predict(model, text)
        result, tuple_re = result_process(list(text), tag_res)
        result = []
        tag = []
        for s, t in tuple_re:
            if t != 'O':
                result.append(s)
                tag.append(t)
        result.append(times[i])
        tag.append('timestamp')
        save_file = dict(zip(tag, result))
        fout.append(save_file)
        if i % 100 == 0:
            print(i)
    out_df = pd.json_normalize(fout)
    out_df.to_csv(save_, index=False)
"""