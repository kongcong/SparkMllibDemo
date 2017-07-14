package com.xj.ml.breeze

import breeze.linalg.{DenseVector, all, any}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/6
  */
object BreezeBooleanFunction {
  def main(args: Array[String]): Unit = {
    val a: DenseVector[Boolean] = DenseVector(true, false, true)
    val b: DenseVector[Boolean] = DenseVector(false, true, true)

    // 与
    val vector1: DenseVector[Boolean] = a :& b
    println(vector1)    // DenseVector(false, false, true)

    // 或
    val vector2: DenseVector[Boolean] = a :| b
    println(vector2)    // DenseVector(true, true, true)

    // 非
    val vector3: DenseVector[Boolean] = !a
    println(vector3)    // DenseVector(false, true, false)

    val vector4: DenseVector[Double] = DenseVector(1.0, 0.0, -2.0)
    println(vector4)    // DenseVector(1.0, 0.0, -2.0)

    // 任意元素非零
    println(any(vector4)) // true

    // 所有元素非零
    println(all(vector4)) // false
  }
}
