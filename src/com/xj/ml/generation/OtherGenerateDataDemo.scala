package com.xj.ml.generation

import org.apache.spark.mllib.util.SVMDataGenerator
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object OtherGenerateDataDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("OtherGenerateDataDemo")
    val sc = new SparkContext(conf)

    // SVM样本生成

    // MFD样本生成

  }
}
