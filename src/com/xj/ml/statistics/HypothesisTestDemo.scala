package com.xj.ml.statistics

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object HypothesisTestDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("PearsonAndSpearman")
    val sc = new SparkContext(conf)


    // 假设检验

    /**
      * Pearson卡方检验(chi-squared,x²) 用于判断拟合度和独立性
      * 不同的输入类型决定了是拟合度检验还是独立性检验
      * 拟合度检验要求输入为Vector，独立性检验要求输入是Matrix
      */

    val v1 = Vectors.dense(43.0, 9.0)
    val v2 = Vectors.dense(44.0, 4.0)
    // 卡方检验
    val c1 = Statistics.chiSqTest(v1, v2)

    println(c1)
    /**
      * Chi squared test summary:
      * method: pearson  统计量
      * degrees of freedom = 1  自由度
      * statistic = 5.482517482517483  值
      * pValue = 0.01920757707591003   概率
      * Strong presumption against null hypothesis: observed follows the same distribution as expected..
      */
  }
}
