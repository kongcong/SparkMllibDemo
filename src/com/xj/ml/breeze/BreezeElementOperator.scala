package com.xj.ml.breeze

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/22
  */
object BreezeElementOperator {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("BreezeElementOperator")
    val sc = new SparkContext(conf)

    val matrix: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))
    println(matrix)
    /**
      * 1.0  2.0  3.0
      * 3.0  4.0  5.0
      */

    // 调整矩阵形状
    val matrix1: DenseMatrix[Double] = matrix.reshape(3, 2)
    println(matrix1)
    /**
     * 1.0  4.0
     * 3.0  3.0
     * 2.0  5.0
     */

    // 矩阵转化成向量
    val vector: DenseVector[Double] = matrix.toDenseVector
    println(vector)
    /**
      * DenseVector(1.0, 3.0, 2.0, 4.0, 3.0, 5.0)
      */

    val matrix2: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    println(matrix2)
    /**
      * 1.0  2.0  3.0
      * 4.0  5.0  6.0
      * 7.0  8.0  9.0
      */

    // 矩阵复制
    val matrix3: DenseMatrix[Double] = matrix2.copy
    println(matrix3)
    /**
      * 1.0  2.0  3.0
      * 4.0  5.0  6.0
      * 7.0  8.0  9.0
      */

    // 取对象线元素
    val diag1: DenseVector[Double] = diag(matrix2)
    println(diag1)
    /**
      * DenseVector(1.0, 5.0, 9.0)
      */

    // 矩阵列赋值
    matrix2(::, 2) := 5.0
    println(matrix2)
    /**
      * 1.0  2.0  5.0
      * 4.0  5.0  5.0
      * 7.0  8.0  5.0
      */

    // 矩阵赋值
    matrix2(1 to 2, 1 to 2) := 5.0
    println(matrix2)
    /**
      * 1.0  2.0  5.0
      * 4.0  5.0  5.0
      * 7.0  5.0  5.0
      */

    val a: DenseVector[Int] = DenseVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    println(a)
    /**
      * DenseVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
      */

    // 子集赋数值
    a(1 to 4) := 5
    println(a)
    /**
      * DenseVector(1, 5, 5, 5, 5, 6, 7, 8, 9, 10)
      */

    // 子集赋向量
    a(1 to 4) := DenseVector(1, 2, 3, 4)
    println(a)
    /**
      * DenseVector(1, 1, 2, 3, 4, 6, 7, 8, 9, 10)
      */

    val a1: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    println(a1)
    /**
      * 1.0  2.0  3.0
      * 4.0  5.0  6.0
      */

    val a2: DenseMatrix[Double] = DenseMatrix((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    println(a2)
    /**
      * 1.0  1.0  1.0
      * 2.0  2.0  2.0
      */

    // 垂直连接矩阵
    val vertcat: DenseMatrix[Double] = DenseMatrix.vertcat(a1, a2)
    println(vertcat)
    /**
      * 1.0  2.0  3.0
      * 4.0  5.0  6.0
      * 1.0  1.0  1.0
      * 2.0  2.0  2.0
      */

    // 横向连接矩阵
    val horzcat: DenseMatrix[Double] = DenseMatrix.horzcat(a1, a2)
    println(horzcat)
    /**
      * 1.0  2.0  3.0  1.0  1.0  1.0
      * 4.0  5.0  6.0  2.0  2.0  2.0
      */

    val b1 = DenseVector(1, 2, 3, 4)
    val b2 = DenseVector(1, 1, 1, 1)
    // 向量连接
    val b3: DenseVector[Int] = DenseVector.vertcat(b1, b2)
    println(b3)
    /**
      * DenseVector(1, 2, 3, 4, 1, 1, 1, 1)
      */
  }
}
