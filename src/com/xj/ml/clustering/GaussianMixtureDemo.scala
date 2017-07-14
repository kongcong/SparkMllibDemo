package com.xj.ml.clustering

import breeze.numerics.{abs, sqrt}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/12
  */
object GaussianMixtureDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GaussianMixtureDemo").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("zhengyue.csv")
    val parsedData: RDD[linalg.Vector] = data.map(s => Vectors.dense(s.trim.split(",").map(_.toDouble)))

    // Cluster the data into two classes using GaussianMixture
    val gmm: GaussianMixtureModel = new GaussianMixture()
      .setK(2).setMaxIterations(1000)
      .run(parsedData)
    val predict: RDD[Int] = gmm.predict(parsedData)

    val datac1: RDD[Double] = parsedData.map(_.toArray(0))   // 每行数据的第一个元素
    val datac2: RDD[Double] = parsedData.map(_.toArray(1))   // 每行数据的第二个元素

    val dataRDD: RDD[(Double, Double)] = datac1.zip(datac2)
    val zip: RDD[((Double, Double), Int)] = dataRDD.zip(predict)
   // zip.foreach(println(_))
    //zip.saveAsTextFile("data2clustering")

    val filter0: RDD[((Double, Double), Int)] = zip.filter(f => f._2 == 0)
    val filter1 = zip.filter(f => f._2 == 1)
    val filter2 = zip.filter(f => f._2 == 2)
    val filter3 = zip.filter(f => f._2 == 3)

    //filter0.foreach(println(_))
    //val c1: RDD[linalg.Vector] = filter1.map(s => Vectors.dense(s._2))\

    val c1 = filter0.map(s => Vectors.dense(s._1._1, s._1._2))

    val stats: MultivariateStatisticalSummary = Statistics.colStats(c1)
    val max: Double = stats.max(0)
    val min: Double = stats.min(0)
    val mean: Double = stats.mean(0)
    val variance: Double = stats.variance(0)
    val sd: Double = sqrt(variance)
    val t = max -min
    val u = (min + max) / 2
    val cp = t / (6 * sd)
    val ca = (mean - u) * 2 / t
    val cpk = cp * (1 - abs(ca))
    val count = c1.count()
    println(s"max = $max")
    println(s"min = $min")
    println(s"mean = $mean")
    println(s"variance = $variance")
    println(s"sd = $sd")
    println(s"t = $t")
    println(s"u = $u")
    println(s"cp = $cp")
    println(s"ca = $ca")
    println(s"cpk = $cpk")
    println(s"count = $count")

//    predict.foreach(println(_))

//    val predict: RDD[Int] = gmm.predict(parsedData)
    //    val map: RDD[Double] = data.map(s => s.trim.split(",")(0).toDouble)
    //
    // Save and load model
    //gmm.save(sc, "modelpath/myGMMModel")
    //val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")


    // output parameters of max-likelihood model
//    for (i <- 0 until gmm.k) {
//      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
//        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
//    }
  }
}
