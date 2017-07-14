package com.xj.ml.recommender

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/14
  */
object ItemCFDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ItemCFDemo").setMaster("local")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据 数据格式 用户id，物品id，评分
    val data = sc.textFile("sample_itemcf2.txt")
    val userdata: RDD[ItemPref] = data.map(_.split(",")).map(f => (ItemPref(f(0), f(1), f(2).toDouble))).cache()

    // 建立模型
    val mysimil = new ItemSimilarity()
    val simil_rdd1 = mysimil.Similarity(userdata, "cooccurrence")
    val recommd: RecommendedItem = new RecommendedItem
    val recommd_rdd1: RDD[UserRecomm] = recommd.Recommend(simil_rdd1, userdata, 30)

    // 打印结果
    println(s"物品相似度矩阵：${simil_rdd1.count()}")
    simil_rdd1.collect().foreach { itemSimi =>
      println(itemSimi.itemid1 + "," + itemSimi.itemid2 + "," + itemSimi.similar)
    }
    /**
      * 物品相似度矩阵：10
      * 2,4,0.3333333333333333
      * 3,4,0.3333333333333333
      * 4,2,0.3333333333333333
      * 3,2,0.3333333333333333
      * 1,2,0.6666666666666666
      * 4,3,0.3333333333333333
      * 2,3,0.3333333333333333
      * 1,3,0.6666666666666666
      * 2,1,0.6666666666666666
      * 3,1,0.6666666666666666
      */

    println(s"用户推荐列表：${recommd_rdd1.count()}")
    recommd_rdd1.collect().foreach { UserRecomm =>
      println(UserRecomm.userid + "," + UserRecomm.itemid + "," + UserRecomm.pref)
    }
    /**
      * 用户推荐列表：11
      * 4,3,0.6666666666666666
      * 4,1,0.6666666666666666
      * 5,4,0.6666666666666666
      * 6,2,0.3333333333333333
      * 6,3,0.3333333333333333
      * 2,4,0.3333333333333333
      * 2,2,1.0
      * 3,1,0.6666666666666666
      * 3,2,0.6666666666666666
      * 1,4,0.3333333333333333
      * 1,3,1.0
      */

  }
}
