package com.xj.ml.dataformats

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object AppendBiasDemo {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("AppendBiasDemo")
    val sc = new SparkContext(conf)

    /**
      * appendBias是对向量增加偏置项，用于回归和分类算法计算中
      *
      * 三个相关数据标记：①样本的真实标记 ②样本在训练集中的标记(可能含噪声)③样本在训练时每次得到的输出标记
      *
      * 偏置(bias)：训练模型的时候，每一次训练得到的训练集预测标签与原始真实标签的偏离程度(即③与①的差)，
      * 如果此偏离程度过小，则会导致过拟合的发生，因为可能将训练集中的噪声也学习了。
      * 所以说偏置刻画了学习算法本身的拟合能力，如果拟合能力不好，偏置较大，出现欠拟合；
      * 反之拟合能力过好，偏置较小，容易出现过拟合。
      * 在训练的时候可以发现这个bias理论上应该是逐渐变小的，表明我们的模型正在不断学习有用的东西。
      * 【当然这是针对只有一个训练集的情况下，如果有多个训练集，就计算出每一个样本在各个训练集下的预测值的均值，
      * 然后计算此均值与真实值的误差即为偏差】
      *
      */

      val context: SQLContext = new SQLContext(sc)
      val vector: linalg.Vector = Vectors.dense(1.0, 2.0)
      val vector1 = MLUtils.appendBias(vector);
      val array: Array[Double] = vector.toArray
      val array1: Array[Double] = vector1.toArray
      for (elem <- array) {
        print(elem + " ")
      }
      println()
      for (elem <- array1) {
        print(elem + " ")
      }
      println()
    /**
      * 打印结果：
      * 1.0 2.0
      * 1.0 2.0 1.0
      */

  }
}
