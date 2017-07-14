package com.xj.ml.clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object KMeansDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("KMeansDemo")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据
    val data = sc.textFile("kmeans_data.txt")

    // 转化数据格式
    val parsedData: RDD[linalg.Vector] = data.map { s =>
      Vectors.dense(s.split(" ").map(_.toDouble))
    }.cache()

    // 新建KMeans聚类模型，并训练
    val initMode = "k-means||"
    val numClusters = 2
    val numIterations = 20
    val model: KMeansModel = new KMeans()
      .setInitializationMode(initMode)
      .setK(numClusters)
      .setMaxIterations(numIterations)
      .run(parsedData)

    // 调用模型预测
    val predict: Int = model.predict(Vectors.dense(9.0, 9.0, 9.0))
    println(s"predict : $predict")   // predict : 0

    // 中心点
    val centers: Array[linalg.Vector] = model.clusterCenters
    println(centers.foreach(println(_)))  // [9.1,9.1,9.1]  [0.1,0.1,0.1]

    // 误差计算
    val WSSSE: Double = model.computeCost(parsedData)
    println(s"Whin Set Sum of Squared Errors = $WSSSE")// Whin Set Sum of Squared Errors = 0.11999999999994547

    // 保存模型
    model.save(sc, "modelpath/kmeansmodel")
    KMeansModel.load(sc, "modelpath/kmeansmodel")

    sc.stop()
  }
}
