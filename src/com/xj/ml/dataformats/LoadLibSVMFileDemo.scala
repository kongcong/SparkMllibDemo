package com.xj.ml.dataformats

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils

/**
  * author : kongcong  
  * number : 27
  * date : 2017/6/21
  */
object LoadLibSVMFileDemo {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("LoadLibSVMFileDemo")
    val sc = new SparkContext(conf)

    /**
      * MLUtils用于辅助、保存、处理MLlib相关算法所需要的数据。
      * 其中最主要的方法是loadLibSVMFile,用于加载LIBSVM格式的数据，返回RDD[LabeledPoint]格式的数据
      * 该数据格式可以用于分类，回归等算法当中
      *
      * 输入LIBSVM数据格式
      * label index1:value1 index2:value2 ...  label代表标签，value1代表特征，index1代表特征位置索引
      *
      * 加载LIBSVM格式的数据 返回RDD[LabelPoint]，LabelPoint格式：
      * (label:Double, features:Vector), label代表标签 features代表特征向量
      *
      */


    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    println(data.take(1))
    // [Lorg.apache.spark.mllib.regression.LabeledPoint;@263da67d 输出结果

  }
}
