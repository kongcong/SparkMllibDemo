package com.xj.ml.generation

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.LogisticRegressionDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object GenerateLogisticRDDDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("GenerateLogisticRDDDemo")
    val sc = new SparkContext(conf)

    /**
      * 这个操作用于生成逻辑回归的训练样本数据，格式为RDD[LabeledPoint]
      * 参数含义：
      * sc : SparkContext
      * 40 : RDD中包括的数据量
      * 3 : 数据维度为三维
      * 1.0 : Epsilon因子
      * 2 : RDD分区数
      * 0.5 : 标签1的概率为0.5
      */
    val logisticRDD: RDD[LabeledPoint] = LogisticRegressionDataGenerator.generateLogisticRDD(sc, 40, 3, 1.0, 2, 0.5)
    println("logisticRDD.count: " + logisticRDD.count())
    val take = logisticRDD.take(5)
    take.foreach(println(_))
    /**
      * 输出结果
      * logisticRDD.count: 40
      * (0.0,[1.1419053154730547,0.9194079489827879,-0.9498666368908959])
      * (1.0,[1.4533448794332902,1.703049287361516,0.5130165929545305])
      * (0.0,[1.0613732338485966,0.9373128243059786,0.519569488288206])
      * (1.0,[1.3931487794809478,1.6410535022701498,0.17945164909645228])
      * (0.0,[1.3558214650566454,-0.8270729973920494,1.6065611415614136])
      */
  }
}
