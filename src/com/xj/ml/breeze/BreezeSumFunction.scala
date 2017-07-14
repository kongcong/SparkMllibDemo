package com.xj.ml.breeze

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, accumulate, sum, trace}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/6
  */
object BreezeSumFunction {
  def main(args: Array[String]): Unit = {
    val a: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    println(a)
    /**
      * 1.0  2.0  3.0
      * 4.0  5.0  6.0
      * 7.0  8.0  9.0
      */

    // 元素求和
    val sum1: Double = sum(a)
    println(sum1)     // 45.0

    // 每一列求和
    val sum2: DenseMatrix[Double] = sum(a, Axis._0)
    println(sum2)    // 12.0  15.0  18.0
    println(sum(a(::, *)))   // 12.0  15.0  18.0

    // 每一行求和
    val sum3: DenseVector[Double] = sum(a, Axis._1)
    println(sum3)    // DenseVector(6.0, 15.0, 24.0)
    println(sum(a(*, ::)))    //DenseVector(6.0, 15.0, 24.0)

    // 对角线元素和
    val trace1: Double = trace(a)
    println(trace1)     // 15.0

    // 累积和 （1,1+2,1+2+3,1+2+3+4）
    println(accumulate(DenseVector(1, 2, 3, 4)))   // DenseVector(1, 3, 6, 10)

  }
}
