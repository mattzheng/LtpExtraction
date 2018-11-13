# LtpExtraction
基于ltp的简单评论观点抽取模块

无监督信息抽取较多都是使用哈工大的ltp作为底层框架。那么基于ltp其实有了非常多的小伙伴进行了尝试，笔者私自将其归纳为：

 - 事件抽取（三元组）
 - 观点抽取

“语言云” 以哈工大社会计算与信息检索研究中心研发的 “语言技术平台（LTP）” 为基础，为用户提供高效精准的中文自然语言处理云服务。 
pyltp 是 LTP 的 Python 封装，提供了分词，词性标注，命名实体识别，依存句法分析，语义角色标注的功能。

 - 技术文档：http://pyltp.readthedocs.io/zh_CN/latest/api.html#id15 
 - 介绍文档：https://www.ltp-cloud.com/intro/#introduction 
 - 介绍文档：http://ltp.readthedocs.io/zh_CN/latest/appendix.html#id5

需要先载入他们训练好的模型，[下载地址](https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=/)

初始化pyltp的时候一定要留意内存问题，初始化任何子模块（`Postagger()` /`NamedEntityRecognizer()`等等）都是需要占用内存，如果不及时释放会爆内存。
之前比较好的尝试是由该小伙伴已经做的小项目：[liuhuanyong/EventTriplesExtraction](https://github.com/liuhuanyong/EventTriplesExtraction)，是做三元组抽取的一个实验，该同学另外一个[liuhuanyong/CausalityEventExtraction](https://github.com/liuhuanyong/CausalityEventExtraction)因果事件抽取的项目也很不错，辛苦写了一大堆规则，之后会对因果推理进行简单描述。

> 笔者也自己写了一个抽取模块，不过只是简单评论观点抽取模块。
> 留心的小伙伴可以基于此继续做很多拓展：搭配用语挖掘，同义词挖掘，新词挖掘
> 笔者的博客连接：[ltp︱基于ltp的无监督信息抽取模块（事件抽取/评论观点抽取）](https://blog.csdn.net/sinat_26917383/article/details/82760214)


----------

# 1 信息抽取 - 搭配抽取

## 1.1 逻辑整理
整个逻辑主要根据依存句法分析，笔者主要利用了以下的关系类型：
![](https://github.com/mattzheng/LtpExtraction/blob/master/pic/001.png)

那么笔者理解 + 整理后得到四类抽取类型：

 - 搭配用语查找（SVB,ATT,ADV）
 - 并列词查找（COO）
 - 核心观点抽取（HED+主谓宾逻辑）
 - 实体名词搭配（词性n ）

其中笔者还加入了停词,可以对结果进行一些筛选。

## 1.2 code粗解读

这边细节会在github上公开，提一下code主要分的内容：`ltp启动模块` / `依存句法解读` / `结果筛选`。

- ltp模块，一定要注意释放模型，不要反复 `Postagger() / Segmentor() / NamedEntityRecognizer() /SementicRoleLabeller()`，会持续Load进内存，然后boom...
- 依存句法模块，笔者主要是整理结果，将其整理为一个dataframe，便于后续结构化理解与抽取内容，可见：
- 结果筛选模块，根据上述的几个关系进行拼接。


> 案例句：艇仔粥料很足，香葱自己添加，很贴心。


![在这里插入图片描述](https://github.com/mattzheng/LtpExtraction/blob/master/pic/002.png)

表的解读，其中：

- word列，就是这句话主要分词结果
- relation列/pos列，代表该词的词性与关系
- match_word列/match_word_n列，根据关系匹配到的词条
- tuples_words列，就是两者贴一起


同时若觉得需要去掉一些无效词搭配，也可以额外添加无效词进来，还是比较弹性的。

## 1.3 结果展示

句子一:

![在这里插入图片描述](https://github.com/mattzheng/LtpExtraction/blob/master/pic/003.png)

句子二：

![在这里插入图片描述](https://github.com/mattzheng/LtpExtraction/blob/master/pic/004.png)

句子三：

![在这里插入图片描述](https://github.com/mattzheng/LtpExtraction/blob/master/pic/005.png)


----------

# 2 LTP的语义角色标注(Semantic Role Labeling,SRL)

更新于20181113

该模块是利用LTP中的SRL模块进行分析


    print(SRLparsing(labeller,words,postags,ToAfter = ['TMP','A1','DIS']))

    ----- 语义角色 -----

    ([['ADV', ('最后', '打')], ['ADV', (['平均', '下来'], '便宜')], ['ADV', ('才', '便宜')], ['A0', ('40', '便宜')]], (True, ['40', '便宜', []]))


与句法模块相似，利用一些组合规则来进行信息抽取,主要以A0为主，A0 - 动作的施事,相当于动作的主体  

此时可以理解为核心主语，然后去找主语的修饰，`TMP(时间),A1(动作的影响),DIS(标记语),PRP(目的)`。

具体可见SRLparsing.py

当然，实际使用的时候,发现会经常报错：

    RuntimeError: CPU memory allocation failed

因为用LTP跑这个耗时 + 耗内存，顶多只是试玩一下，不太利用用于大批量操作。









