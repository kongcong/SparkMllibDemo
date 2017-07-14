package com.xj.ml.generation

import org.apache.spark.mllib.util.KMeansDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object GenerateKMeansRDDDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("GenerateKMeansRDDDemo")
    val sc = new SparkContext(conf)

    /**
      * 随机生成40个样本，数据维度为三维，聚类中心数为5，初始中心分布的缩放因子为1.0（袁凡说可能是初始中心点的选取范围），RDD的分区数为2
      */
    val KMeansRDD: RDD[Array[Double]] = KMeansDataGenerator.generateKMeansRDD(sc, 40, 5, 3, 1.0, 2)

    println("KMeansRDD.count: " + KMeansRDD.count())
    val take: Array[Array[Double]] = KMeansRDD.take(5)
    take.foreach(_.foreach(println(_)))

    /**
      * 计算结果：
      * KMeansRDD.count: 40
      *
      * 2.2838106309461095
      * 1.8388158979655758
      * -1.8997332737817918
      * -0.6536454069660477
      * 0.9840269254342955
      * 0.19763938858718594
      * 0.24415182644986977
      * -0.4593305783720648
      * 0.3286249752173309
      * 1.8793621718715983
      * 1.4433606519575122
      * -0.9420612755690412
      * 2.7663276890005077
      * -1.4673057796056233
      * 0.39691668230812227
      *
      */

  }
}
