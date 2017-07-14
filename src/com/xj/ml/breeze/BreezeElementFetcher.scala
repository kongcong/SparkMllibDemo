package com.xj.ml.breeze

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object BreezeElementFetcher {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("BreezeElementFetcher")
    val sc = new SparkContext(conf)

    val a: DenseVector[Int] = DenseVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    println(a(0))  // 1

    println(a(1 to 4))  // DenseVector(2, 3, 4, 5)

    println(a(5 to 0 by -1))  // DenseVector(6, 5, 4, 3, 2, 1)

    println(a(1 to -1))  // DenseVector(2, 3, 4, 5, 6, 7, 8, 9, 10)

    println(a(-1))  // 10  -1表示最后一个元素

    val matrix: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))
    println(matrix)
    /**
      * 1.0  2.0  3.0
      * 3.0  4.0  5.0
      */

    println(matrix(0, 1))   // 2.0
    println(matrix(::, 1))  // DenseVector(2.0, 4.0)  // 矩阵指定列

  }
}
