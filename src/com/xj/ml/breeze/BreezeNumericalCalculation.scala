package com.xj.ml.breeze

import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/6
  */
object BreezeNumericalCalculation { // Breeze数值计算函数
  def main(args: Array[String]): Unit = {
    val a: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val b: DenseMatrix[Double] = DenseMatrix((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))

    // 元素加法（对应元素相加）
    val matrix1: DenseMatrix[Double] = a + b
    println(matrix1)
    /**
      * 2.0  3.0  4.0
      * 6.0  7.0  8.0
      */

    // 元素乘法（对应元素相乘）
    val matrix2: DenseMatrix[Double] = a :* b
    println(matrix2)
    /**
      * 1.0  2.0   3.0
      * 8.0  10.0  12.0
      */

    // 元素除法（对应元素相除）
    val matrix3: DenseMatrix[Double] = a :/ b
    println(matrix3)
    /**
      * 1.0  2.0  3.0
      * 2.0  2.5  3.0
      */

    // 对应位置元素比较大小
    val matrix4: DenseMatrix[Boolean] = a :< b
    println(matrix4)
    /**
      * false  false  false
      * false  false  false
      */

    // 对应位置元素比较是否相等
    val matrix5: DenseMatrix[Boolean] = a :== b
    println(matrix5)
    /**
      * true   false  false
      * false  false  false
      */

    // 对应位置元素加上指定数值
    val matrix6: DenseMatrix[Double] = a :+= 1.0
    println(matrix6)
    /**
      * 2.0  3.0  4.0
      * 5.0  6.0  7.0
      */

    // 对应位置元素乘上指定数值
    val matrix7: DenseMatrix[Double] = a :*= 2.0
    println(matrix7)
    /**
      * 4.0   6.0   8.0
      * 10.0  12.0  14.0
      */

    // 元素最大值
    val max1: Double = max(a)
    println(max1)    // 14.0

    // 元素最大值的位置
    val argmax1: (Int, Int) = argmax(a)
    println(argmax1)    // (1,2)

    // 向量点积(1*1 + 2*1 + 3*1 + 4*1)
    val i: Int = DenseVector(1, 2, 3, 4) dot(DenseVector(1, 1, 1, 1))
    println(i)    // 10

  }
}
