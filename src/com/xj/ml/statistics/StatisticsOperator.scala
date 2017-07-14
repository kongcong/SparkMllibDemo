package com.xj.ml.statistics

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object StatisticsOperator {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("StatisticsOperator")
    val sc = new SparkContext(conf)

    // 列统计汇总

    val data = sc.textFile("sample_stat.txt")
      .map(_.split(","))
      .map(f => f.map(f => f.toDouble))
    val data1 = data.map(f => Vectors.dense(f))
    val stats = Statistics.colStats(data1)

    // 每一列的最大值
    println("max:" + stats.max)
    // 每一列的最小值
    println("min:" + stats.min)
    // 每一列的平均数
    println("mean:" + stats.mean)
    // 每一列的方差
    println("variance:" + stats.variance)
    // 每一列的L1范数  L1范数是指向量中各个元素绝对值之和
    println("normL1:" + stats.normL1)
    // 过拟合是模型训练时候的误差很小，但在测试的时候误差很大，
    // 也就是我们的模型复杂到可以拟合到我们的所有训练样本了，
    // 但在实际预测新的样本的时候，糟糕的一塌糊涂
    // L2范数 ： 在回归里面，有人把有它的回归叫“岭回归”（Ridge Regression）解决过拟合  L2范数是指向量各元素的平方和然后求平方根
    // 每一列的L2范数
    println("normL2:" + stats.normL2)

  }
}
