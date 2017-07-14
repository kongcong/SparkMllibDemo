package com.xj.ml.classification

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object RandomForestDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("RandomForestDemo")
    val sc = new SparkContext(conf)

    // 加载数据
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
    // 切分训练集和测试集
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))


    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()  // 空categoricalFeaturesInfo表明所有特征是连续的
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    // 训练随机森林模型
    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // 评估模型，计算误差
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)

    // 保存和加载模型
    model.save(sc, "modelpath/RandomForestModel")
    //val sameModel = RandomForestModel.load(sc, "modelpath/RandomForestModel")

  }
}
