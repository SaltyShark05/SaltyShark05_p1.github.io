##Location Predict 论文模型和Attention is all you need 论文理解
- 关于Transformer的学习理解
    - 推荐csdn上的一篇文章[自然语言处理之Attention大详解](https://blog.csdn.net/wuzhongqiang/article/details/104414239?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166971993516782412567146%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166971993516782412567146&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-104414239-null-null.142^v67^js_top,201^v3^add_ask,213^v2^t3_esquery_v1&utm_term=Attention%20Is%20All%20You%20Need&spm=1018.2226.3001.4187)
    - 还有b站上一位up讲的[Self-Attention详解](https://www.bilibili.com/video/BV1ht4y187JE/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=a508897fc8071216af1f20ee67641a65)
    - 在配合论文中内容（的翻译）的帮助，对于模型有了一些浅显的理解
- 关于Pytorch的学习
    - 在pytorch的安装中借鉴了[pytorch安装详解](https://blog.csdn.net/qq_32863549/article/details/107698516?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166990838416800213059864%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166990838416800213059864&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-107698516-null-null.142^v67^js_top,201^v3^add_ask,213^v2^t3_esquery_v1&utm_term=pytorch%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)
    - 在b站上up我是土堆的[pytorch教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click)
    - 还有我认为最重要的一点：就是善用搜索引擎或是pytorch中自带的函数说明书，然后再另起一个文件来进行测试，这样可以加深对函数的理解（纠正错误的函数理解0.0）
- 在创建Location Predict论文模型时
    - 其实很多模块在论文中已经帮你划分好了，只需要用代码来实现公式中的内容就好，当然也要理解那部分模块的用途，这样函数在后面的模块中有复用的话会更容易操作一些
    - 不过既然用Pytorch这个工具来辅助你进行模型的搭建，有很多操作就不必拘泥于论文中所给的公式了
        - 如relu函数
        - ```python
            # 前馈神经网络
            def feed_forward(m, w1, b1, w2, b2):
                relu = torch.nn.ReLU(inplace=True)
                m_dot = torch.mm(relu(torch.mm(m, w1) + b1), w2) + b2
                return m_dot
            ```
        - 如softmax函数
        - ```python
            # 单头自注意力
            def self_attention(s, dk):
                relation = torch.mm(s, s.T) / dk ** 0.5
                selfattn_s = torch.mm(torch.nn.functional.softmax(relation), s)
                return selfattn_s
            ```
        - 最想说的就是在论文4.3中实现的subword feature：因为实际上就是对于输入的c矩阵进行一次卷积操作和一次池化操作，就可以直接用pytorch中集成好的nn.Conv2d函数和nn.MaxPool2d函数（绝不是偷懒哈），可以更加简化代码的内容
        - ```python
            # 子字功能
            def subword_feature(c, k, m_character_length, f_index_filter, w1, b1, w2, b2, dk, h, w):
                conv2d = torch.nn.Conv2d(1, f_index_filter, 2 * k + 1, stride=1)
                h_conv = conv2d(c)
                maxpool2d = torch.nn.MaxPool2d(2 * k, 2 * k - 1)
                h_pool = maxpool2d(h_conv)
            ```
    - 偷偷说一下，那个在群里问数据集在哪的憨憨就是我哈_(•̀ω•́ 」∠)_ 因为我觉得如果要进行训练的话，可能加入数据集之后（因为自己找的那个数据集和论文里的不太一样）就要对函数的输入还有函数内部的一些操作进行一些改动，就会很麻烦。只完成模型的话也是变向的解决了我对那个奇怪数据集的问题吧ovo
- 仓库链接：https://github.com/SaltyShark05/Yilu_Stdio_tasks.github.io