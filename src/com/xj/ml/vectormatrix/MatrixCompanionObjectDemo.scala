package com.xj.ml.vectormatrix

import org.apache.spark.mllib.linalg.{Matrices, Matrix, SparseMatrix, Vectors}

import scala.util.Random

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/10
  */
object MatrixCompanionObjectDemo {
  def main(args: Array[String]): Unit = {

    /**
      * 创建密集矩阵
      *
      * @param numRows 行数
      * @param numCols 列数
      * @param values 矩阵值
      */
    val dense: Matrix = Matrices.dense(3, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0, 0, 0))
    println(dense)
    /**
      * 1.0  4.0  0.0
      * 2.0  5.0  0.0
      * 3.0  6.0  0.0
      */


    /**
      * 创建稀疏矩阵
      *
      * @param numRows 行数
      * @param nomcols 列数
      * @param colPtr 列标识(切分点)
      * @param rowIndices 行索引
      * @param values 非零值
      *
      */
    val sparse = Matrices.sparse(3, 3, Array(0, 2, 3, 6),Array(0, 2, 1, 0, 1, 2), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    println(sparse)
    /**
      * 3 x 3 CSCMatrix
      * (0,0) 1.0
      * (2,0) 2.0
      * (1,1) 3.0
      * (0,2) 4.0
      * (1,2) 5.0
      * (2,2) 6.0
      */


    // 创建全0矩阵
    val zeros: Matrix = Matrices.zeros(3, 4)
    println(zeros)
    /**
      * 0.0  0.0  0.0  0.0
      * 0.0  0.0  0.0  0.0
      * 0.0  0.0  0.0  0.0
      */

    // 创建全1矩阵
    val ones: Matrix = Matrices.ones(3, 4)
    println(ones)
    /**
      * 1.0  1.0  1.0  1.0
      * 1.0  1.0  1.0  1.0
      * 1.0  1.0  1.0  1.0
      */

    // 创建单位矩阵
    val eye: Matrix = Matrices.eye(4)
    println(eye)
    /**
      * 1.0  0.0  0.0  0.0
      * 0.0  1.0  0.0  0.0
      * 0.0  0.0  1.0  0.0
      * 0.0  0.0  0.0  1.0
      */

    // 创建稀疏单位矩阵
    val speye: _root_.org.apache.spark.mllib.linalg.SparseMatrix = SparseMatrix.speye(4)
    println(speye)
    /**
      * 4 x 4 CSCMatrix
      * (0,0) 1.0
      * (1,1) 1.0
      * (2,2) 1.0
      * (3,3) 1.0
      */

    // 创建随机矩阵 平均分布
    val rand: Matrix = Matrices.rand(3, 4, Random.self)
    println(rand)
    /**
      * 0.7484436482765457  0.5033737185553904  0.5267851018155542   0.9403358531108571
      * 0.6712223143751761  0.7026210478024245  0.7318965440586115   0.7290552064931253
      * 0.5374985358138044  0.7170708802486215  0.11116596576945015  0.006126792223793309
      */

    // 创建随机稀疏矩阵，平均分布
    val sprand: Matrix = Matrices.sprand(3, 3, 0.5, Random.self)
    println(sprand)
    /**
      * 3 x 3 CSCMatrix
      * (0,0) 0.5404482923878735
      * (1,0) 0.05457661839998651
      * (0,1) 0.438460485043879
      * (1,1) 0.8743642300868616
      * (1,2) 0.13838453177746923
      */
    
    // 创建随机矩阵，正态分布
    val randn: Matrix = Matrices.randn(3, 4, Random.self)
    println(randn)
    
    // 创建随机稀疏矩阵，正态分布
    val sprandn: Matrix = Matrices.sprandn(3, 4, 0.2, Random.self)
    println(sprandn)
    
    // 创建对角矩阵
    val diag: Matrix = Matrices.diag(Vectors.dense(1, 2, 3, 4))
    println(diag)
    
    // 横向连接矩阵
    val m1 = Matrices.dense(3, 3, Array(1, 2, 3, 4, 5, 6, 7, 8, 9))
    val m2 = Matrices.dense(3, 3, Array(1, 1, 1, 1, 1, 1, 1, 1, 1))
    val horzcat: Matrix = Matrices.horzcat(Array(m1, m2))
    println(horzcat)
    /**
      * 1.0  4.0  7.0  1.0  1.0  1.0
      * 2.0  5.0  8.0  1.0  1.0  1.0
      * 3.0  6.0  9.0  1.0  1.0  1.0
      */

    // 垂直连接矩阵
    val vertcat: Matrix = Matrices.vertcat(Array(m1, m2))
    println(vertcat)
    /**
      * 1.0  4.0  7.0
      * 2.0  5.0  8.0
      * 3.0  6.0  9.0
      * 1.0  1.0  1.0
      * 1.0  1.0  1.0
      * 1.0  1.0  1.0
      */

  }
}
