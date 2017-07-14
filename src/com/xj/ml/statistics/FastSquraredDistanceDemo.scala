package com.xj.ml.statistics

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object FastSquraredDistanceDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("FastSquraredDistanceDemo")
    val sc = new SparkContext(conf)

    val data: RDD[Array[Double]] = sc.textFile("sample_stat.txt")
      .map(_.split(","))
      .map(f => f.map(f => f.toDouble))
    val data1: RDD[linalg.Vector] = data.map(f => Vectors.dense(f))

   /* val norms = data1.map(norm(_, 2.0))
    val zippedData = data1.zip(norms).map{ case( v,norm) =>
       new VectorWithNorm(v,norms)
    }*/

    /**
      * fastSquaredDistance方法是一种快速计算向量距离的方法
      * 主要用于KMeans聚类算法
      *
      * 返回向量之间的平方欧式距离。
      * 在两个向量的范数已经给定的情况下，比直接计算快
      * 特别是包含稀疏向量时，计算效率更高
      *
      */
      //KMeans.fastSquaredDistance(v1, v2)







  }
}

