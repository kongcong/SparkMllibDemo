package com.xj.ml.clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA, LDAModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object LDADemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("LDADemo")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    /**
      * 加载数据 数据格式为： 词频1 词频2 词频3....
      *
      * 返回的数据格式为document:RDD[(Long, Vector)]
      * 其中Long为文章ID，Vector为文章分词后的词向量
      * 可以读取指定目录下的数据，通过分词及数据格式的转换，转换成RDD[(Long,Vector)]即可
      */
    val data: RDD[String] = sc.textFile("sample_lda_data.txt")
    val parsedData: RDD[linalg.Vector] = data.map(s => Vectors.dense(s.trim.split(" ").map(_.toDouble)))
    // 给文档增加唯一ID
    val corpus: RDD[(Long, linalg.Vector)] = parsedData.zipWithIndex().map(_.swap).cache()

    // 建立模型，设置训练参数，训练模型
    val ldaModel: LDAModel = new LDA()
      .setK(3) // 主题数量
      .setDocConcentration(5) // 超参alpha 值越大意味着越平滑（更正规化）
      .setTopicConcentration(5) // 超参beta 对于主题的词分布通常叫Beta
      .setMaxIterations(20) // 迭代次数
      .setSeed(0L) // 随机种子
      .setCheckpointInterval(10) // 检查间隔(默认10，当节点失败时，可以通过检查点进行数据恢复。通过检查点也有助于消除磁盘中的临时shuffle文件)
      .setOptimizer("em") // LDA求解的优化算法 目前支持的优化计算类型有：em(必须 > 1.0)  online(必须 >= 0)
      .run(corpus)

    // 模型输出，模型参数输出，结果输出
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    // Learned topics (as distributions over vocab of 11 words):

    // 主题分布
    val topics: Matrix = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) {
        print(" " + topics(word, topic))
      }
      println()
    }
    /**
      * Topic 0: 7.437585629255351 7.21570602341692 3.612574815389632 17.963021799137596 5.7077961447107 4.961421660866798 12.731935474399801 2.275958762916907 3.0887503720183687 8.06210050051863 18.77443443432135
      * Topic 1: 9.745728853344257 12.412981990213032 4.309548961176666 8.223662528504065 10.63142922715975 9.488661743113443 8.816358888018451 4.298660236177567 2.4872253252683105 8.28101031631224 6.041328735450762
      * Topic 2: 8.816685517400394 9.37131198637005 4.0778762234337025 13.813315672358339 8.660774628129548 7.549916596019761 9.451705637581746 3.4253810009055266 2.424024302713321 7.656889183169128 8.184236830227892
      */

    // 主题分布排序
    ldaModel.describeTopics(4)

    // 文档分布
    val distLDAModel: DistributedLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]

    println("文档分布：")
    distLDAModel.topicDistributions.foreach(println(_))
    /**
      * 文档分布：
      * (8,[0.23953838371693492,0.41154391910948157,0.3489176971735835])
      * (10,[0.43149754416519387,0.23556758034173508,0.33293487549307105])
      * (4,[0.41029489315898443,0.2477609080392831,0.34194419880173244])
      * (11,[0.20979467588762835,0.4537375364118028,0.33646778770056884])
      * (0,[0.2979553770395885,0.3739169154377783,0.3281277075226332])
      * (1,[0.27280146347774675,0.39084864123938434,0.336349895282869])
      * (6,[0.5316139195059199,0.2059705919033965,0.26241548859068353])
      * (3,[0.5277762550221997,0.20882605277709115,0.2633976922007093])
      * (9,[0.27482666043742815,0.41148754032514917,0.3136857992374226])
      * (7,[0.42464610239585504,0.2380770679571216,0.3372768296470233])
      * (5,[0.24464389209216816,0.40747788804339075,0.3478782198644412])
      * (2,[0.2973287069168619,0.3780115877202355,0.32465970536290256])
      *
      */

  }
}
