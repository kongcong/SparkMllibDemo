package com.xj.ml.statistics

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object PearsonAndSpearmanOperator {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("PearsonAndSpearman")
    val sc = new SparkContext(conf)

    // 相关系数

    val data = sc.textFile("sample_stat.txt")
      .map(_.split(","))
      .map(f => f.map(f => f.toDouble))
    val data1 = data.map(f => Vectors.dense(f))

    // Pearson相关系数表示两个数值变量的线性相关性，一般适用于正态分布。
    // 取值范围是[-1, 1] 取值为0表示不相关，取值为0~-1表示负相关 取值0~1表示正相关
    // Spearman相关系数也用来表达两个变量的相关性 但没有Pearson相关系数对变量的分布那么严格
    // 另外Spearman相关系数可以更好的用于测度变量的排序关系

    // 计算Pearson相关系数 Spearman相关系数
    val corr1 = Statistics.corr(data1, "pearson")
    val corr2 = Statistics.corr(data1, "spearman")

    val x1 = sc.parallelize(Array(1.0, 2.0, 3.0, 4.0))
    val y1 = sc.parallelize(Array(5.0, 6.0, 6.0, 6.0))
    val corr3 = Statistics.corr(x1, y1, "pearson")

    println("corr1:" + corr1)
    println("corr2:" + corr2)
    println("corr3:" + corr3)
  }
}
