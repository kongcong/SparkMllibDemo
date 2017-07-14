package com.xj.ml.breeze

import breeze.linalg.DenseVector
import breeze.numerics._

import scala.collection.mutable

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/6
  */
object BreezeRoundFunction {
  def main(args: Array[String]): Unit = {
    val a: DenseVector[Double] = DenseVector(1.2, 0.6, -2.3)
    println(a)      // DenseVector(1.2, 0.6, -2.3)

    // 四舍五入
    val round1: DenseVector[Long] = round(a)
    println(round1) // DenseVector(1, 1, -2)

    // 向上取整
    val ceil1: DenseVector[Double] = ceil(a)
    println(ceil1)  // DenseVector(2.0, 1.0, -2.0)

    // 向下取整
    val floor1: DenseVector[Double] = floor(a)
    println(floor1) // DenseVector(1.0, 0.0, -3.0)

    // 符号函数
    val signum1: DenseVector[Double] = signum(a)
    println(signum1)// DenseVector(1.0, 1.0, -1.0)

    // 求绝对值
    val abs1: DenseVector[Double] = abs(a)
    println(abs1)   // DenseVector(1.2, 0.6, 2.3)


  }
}
