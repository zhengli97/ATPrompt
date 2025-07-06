
# Details of the attribute search process.

In Readme.md, we provide instructions for the attribute search method, and in Github Releases, we provide logs of attribute search on ten datasets (Caltech101, OxfordPets, Stanford Cars, Flowers-102, Food-101, FGVC Aircraft, SUN 397, DTD, EuroSAT, UCF-101) for your reference.

If you are interested in this process, the following key information or experience can be used as a reference for readers to reproduce or use it **faster**:

(1) During the search (training) process, we set the epoch to 40. In order to speed up, it can be set to 20 or even 10. Such settings will not have a significant impact on the results, but will be faster. We set it to 40 for the stability of training.

(2) For multiple attribute bases, there will be a large number of combined attributes for search due to the combination. When the training data is large, the search process will become complicated and very slow. In order to speed up, you can manually filter out attribute groups with more than 3 attributes, which can greatly speed up. The reason is that we conducted experiments on 11 data sets and found that the appropriate attribute groups are roughly between 1 and 3. Attribute groups greater than 4 often have very low confidence and are of little practical significance. If you want to speed up further, you can more aggressively filter out attribute groups greater than or equal to 3.

(3) When the training data is still too large to be processed, you can split the attribute groups, search them separately, and then merge them for search. For example: when 20 attribute groups are stuffed into the GPU together, the GPU memory will overflow. At this time, you can divide them into groups of 10, get the top five with the largest weights in the two groups, and then search again.

(4) When searching, we will choose shots=16. If you want to speed up further, you can set it to shots=8 or even shots=4.

<br>

# 中文版

在Readme.md中，我们提供了要去进行属性搜索方法的指引，并在Github Releases中提供了在十个数据集(Caltech101, OxfordPets, Stanford Cars, Flowers-102, Food-101, FGVC Aircraft, SUN 397, DTD, EuroSAT, UCF-101)上进行属性搜索的日志，供大家参考。

如果你对这个过程感兴趣，以下有一些关键信息或者经验可以供读者参考，更快的进行复现或使用：

(1) 在搜索（训练）过程中，我们将epoch设定为40，为了加速，可以设定成20，甚至10，这样的设定不会对结果产生明显影响，反而会更快，我们设定成40是为了训练的稳定。

(2) 对于多个属性基，会因为组合的原因产生非常多种组合属性用于搜索，当训练数据比较大时，搜索过程会变复杂且非常慢，为了加速，可以手动过滤掉属性大于3的属性组，这能够大大加速。原因是，我们在11个数据集上进行了实验发现合适的属性组大致在1-3之间，大于4之后的属性组往往置信度非常低，实际的意义不大。如果想要进一步加速，可以更加激进的选择过滤掉大于等于3的属性组。

(3) 当训练数据还是太大，难以进行的时候，可以对属性组进行切分，分别搜索，然后再合并搜索。举个例子：有20个属性组一起塞进GPU里的时候会显存溢出，这时候可以划分成10个为一组，分别得到两组的权重最大的前五个，然后再进行搜索。

(4) 在搜索时，我们会选择shots=16，如果想要进一步加速可以设定为shots=8，甚至shots=4。

