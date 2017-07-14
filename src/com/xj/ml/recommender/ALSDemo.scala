package com.xj.ml.recommender

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/12
  */
object ALSDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ALSDemo").setMaster("local")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    /**
      * 读取样本数据
      *
      * 数据格式为：用户id，物品id，评分
      *
      * 1,1,5.0
      * 2,4,1.0
      * 3,1,1.0
      * 4,4,5.0
      */
    val data: RDD[String] = sc.textFile("als/test.data")
    val ratings: RDD[Rating] = data.map(_.split(",") match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })

    val rank = 10
    val numIterations = 20
    /**
      * 训练模型
      */
    val model: MatrixFactorizationModel = ALS.train(ratings, rank, numIterations, 0.01)

    // 预测结果
    val usersProducts: RDD[(Int, Int)] = ratings.map {
      case Rating(user, product, rate) =>
        (user, product)
    }
    val predictions: RDD[((Int, Int), Double)] = model.predict(usersProducts).map {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds: RDD[((Int, Int), (Double, Double))] = ratings.map {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions)

    ratesAndPreds.foreach { f =>
      println(f._1._1, f._1._2, f._2._1, f._2._2)
    }
    /**
      * (4,4,5.0,4.994574009576763)
      * (1,4,1.0,1.0001571199897574)
      * (1,1,5.0,4.995796739817866)
      * (3,1,1.0,1.000401704679625)
      * (4,2,5.0,4.994574009576763)
      * (2,2,1.0,1.0001571199897574)
      * (2,3,5.0,4.995796739817866)
      * (4,1,1.0,1.000401704679625)
      * (2,4,1.0,1.0001571199897574)
      * (1,2,1.0,1.0001571199897574)
      * (3,2,5.0,4.994574009576763)
      * (3,4,5.0,4.994574009576763)
      * (3,3,1.0,1.000401704679625)
      * (4,3,1.0,1.000401704679625)
      * (2,1,5.0,4.995796739817866)
      * (1,3,5.0,4.995796739817866)
      */

    val MSE = ratesAndPreds.map {
      case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error = " + MSE)  // Mean Squared Error = 1.1823705393147908E-5

    // 保存与加载模型
    //model.save(sc, "modelpath/alsmodel")
    //MatrixFactorizationModel.load(sc, "modelpath/alsmodel")


  }
}
