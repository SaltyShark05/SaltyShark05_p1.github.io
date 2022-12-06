import torch
import math


# 单头自注意力
def self_attention(s, dk):
    relation = torch.mm(s, s.T) / dk ** 0.5
    selfattn_s = torch.mm(torch.nn.functional.softmax(relation), s)
    return selfattn_s


# 多头自注意力
def mult_self_attention(s, dk, h, w):
    d_new = dk // h
    m = torch.tensor([], requires_grad=False)
    for i in range(0, h-1):
        w_new = w[i, :, :]
        s_new = torch.mm(s, w_new)
        m_new = self_attention(s_new, d_new)
        m = torch.cat([m, m_new], 0)
    m = torch.add(m, s)
    return m


# 前馈神经网络
def feed_forward(m, w1, b1, w2, b2):
    relu = torch.nn.ReLU(inplace=True)
    m_dot = torch.mm(relu(torch.mm(m, w1) + b1), w2) + b2
    return m_dot


# 单步位置编码
def one_step_pos_encoding(w, e_position, i):
    w = w + e_position[i, :]
    return w


# 位置编码组合
def pos_encoding(w, e_position, n):
    for i in range(0, n-1):
        w = one_step_pos_encoding(w, e_position, i)
    return w


# 单词表示
def text_representation(m_dot, n):
    v_text = torch.tensor([], requires_grad=False)
    for i in range(0, n-1):
        v_text = v_text + m_dot[i, :]
    return v_text


# 子字功能
def subword_feature(c, k, m_character_length, f_index_filter, w1, b1, w2, b2, dk, h, w):
    conv2d = torch.nn.Conv2d(1, f_index_filter, 2 * k + 1, stride=1)
    h_conv = conv2d(c)
    maxpool2d = torch.nn.MaxPool2d(2 * k, 2 * k - 1)
    h_pool = maxpool2d(h_conv)
    m_char = mult_self_attention(h_pool, dk, h, w)
    m_dot_char = feed_forward(m_char, w1, b1, w2, b2)
    v_char = torch.tensor([], requires_grad=False)
    for i in range(1, m_character_length / (2 * k - 1)):
        v_char = v_char + m_dot_char[i, :]
    return v_char


# 联合训练
def joint_training(v_text, v_char, w_city, w_country):
    v_tweet = torch.cat([v_text, v_char], 0)
    v_city = torch.mm(v_tweet, w_city)
    v_country = torch.mm(v_tweet, w_country)
    p_city = torch.nn.functional.softmax(v_city)
    p_country = torch.nn.functional.softmax(v_country)
    y_dot_city = torch.argmax(p_city, dim=0)
    y_dot_country = torch.argmax(p_country, dim=0)
    # 损失函数
    loss_city = math.log(p_city[y_dot_city])
    loss_country = math.log(p_country[y_dot_country])
    loss = - loss_city - loss_country
    return y_dot_city, y_dot_country, loss
