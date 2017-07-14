package com.xj.ml.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object DecisionTreeDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("DecisionTreeDemo")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据（libSVM格式）
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
    // 切分训练集和测试集
    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))


    val numClasses = 2   // 分类数量
    val categoricalFeaturesInfo = Map[Int, Int]()  // 用map存储类别（离散）特征及每个类别特征对应值（类别）的数量
    val impurity = "gini"   // 纯度计算方法
    val maxDepth = 5        // 树的最大高度 建议值5
    val maxBins = 32        // 用于分裂特征的最大划分数量 建议值32
    /**
      * 新建决策树
      */
    val model: DecisionTreeModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // 误差计算
    val labelAndPreds: RDD[(Double, Double)] = testData.map { point =>
      val prediction: Double = model.predict(point.features)
      (point.label, prediction)
    }
    val print_predict: Array[(Double, Double)] = labelAndPreds.take(20)
    println("label" + "\t" + "prediction")
    for (i <- 0 to print_predict.length - 1)
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    /**
      * label	prediction
      * 1.0	  1.0
      * 1.0	  1.0
      * 0.0	  0.0
      * 0.0	  0.0
      * 0.0	  0.0
      * 1.0	  1.0
      * 1.0	  1.0
      * 0.0 	0.0
      * 1.0	  1.0
      * 0.0 	0.0
      */

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("learned classification tree model:\n" + model.toDebugString)
    /**
      * learned classification tree model:
      * DecisionTreeModel classifier of depth 2 with 5 nodes
      *   If (feature 434 <= 0.0)
      *    If (feature 100 <= 165.0)
      *     Predict: 0.0
      *    Else (feature 100 > 165.0)
      *     Predict: 1.0
      *   Else (feature 434 > 0.0)
      *   Predict: 1.0
      */

    // 保存与加载模型
    val modelPath = "modelPath/DecisionTreeModel"
    model.save(sc, modelPath)
   // DecisionTreeModel.load(sc, modelPath)

  }
}
