package com.xj.ml.classification

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/11
  */
object GradientBoostedTreesDemo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("GradientBoostedTreesDemo")
    val sc = new SparkContext(conf)

    // 加载数据
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
    // 切分训练集和测试集
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // 使用LogLoss作为默认参数
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3    // 实践中使用增加迭代次数
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    // categoricalFeaturesInfo为空表示所有特征是连续的
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    // 训练GBDT模型
    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // 评估模型，计算误差
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)    // Test Error = 0.06451612903225806
    println("Learned classification GBT model:\n" + model.toDebugString)
    /**
      * Learned classification GBT model:
      * TreeEnsembleModel classifier with 3 trees

      *   Tree 0:
      *    If (feature 406 <= 0.0)
      *      Predict: -1.0
      *     Else (feature 406 > 0.0)
      *      Predict: 1.0
      *   Tree 1:
      *     If (feature 406 <= 0.0)
      *     If (feature 183 <= 228.0)
      *       Predict: -0.4768116880884702
      *      Else (feature 183 > 228.0)
      *       Predict: -0.4768116880884703
      *     Else (feature 406 > 0.0)
      *      If (feature 404 <= 31.0)
      *       Predict: 0.47681168808847024
      *      Else (feature 404 > 31.0)
      *       Predict: 0.47681168808846996
      *   Tree 2:
      *    If (feature 406 <= 0.0)
      *      If (feature 511 <= 0.0)
      *      Predict: -0.4381935810427206
      *     Else (feature 511 > 0.0)
      *       Predict: -0.4381935810427206
      *     Else (feature 406 > 0.0)
      *      If (feature 322 <= 27.0)
      *      Predict: 0.4381935810427206
      *      Else (feature 322 > 27.0)
      *       Predict: 0.43819358104272066
      */

    // 保存和加载模型
    model.save(sc, "modelpath/gdbtmodel")
    val sameModel = GradientBoostedTreesModel.load(sc, "modelpath/gdbtmodel")

  }
}
