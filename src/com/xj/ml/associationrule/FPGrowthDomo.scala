package com.xj.ml.associationrule

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/12
  */
object FPGrowthDomo {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("FPGrowthDomo").setMaster("local")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据
    val data: RDD[String] = sc.textFile("sample_fpgrowth.txt")
    val examples: RDD[Array[String]] = data.map(_.split(" "))

    // 建立模型
    val minSupport = 0.2   // 最小支持度
    val numPartition = 10  // 设置分区数 默认为输入样本数据的分区数
    val model: FPGrowthModel[String] = new FPGrowth()
      .setMinSupport(minSupport)
      .setNumPartitions(numPartition)
      .run(examples)

    // 输出结果
    println(s"Number of frequent itemsets : ${model.freqItemsets.count()}")
    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + "," + itemset.freq)
    }
    /**
      * Number of frequent itemsets : 54
      * [z],5
      * [x],4
      * [x,z],3
      * [y],3
      * [y,x],3
      * [y,x,z],3
      * [y,z],3
      * [r],3
      * [r,x],2
      * [r,z],2
      * [s],3
      * [s,y],2
      * [s,y,x],2
      * [s,y,x,z],2
      * [s,y,z],2
      * [s,x],3
      * [s,x,z],2
      * [s,z],2
      * [t],3
      * [t,y],3
      * [t,y,x],3
      * [t,y,x,z],3
      * [t,y,z],3
      * [t,s],2
      * [t,s,y],2
      * [t,s,y,x],2
      * [t,s,y,x,z],2
      * [t,s,y,z],2
      * [t,s,x],2
      * [t,s,x,z],2
      * [t,s,z],2
      * [t,x],3
      * [t,x,z],3
      * [t,z],3
      * [p],2
      * [p,r],2
      * [p,r,z],2
      * [p,z],2
      * [q],2
      * [q,y],2
      * [q,y,x],2
      * [q,y,x,z],2
      * [q,y,z],2
      * [q,t],2
      * [q,t,y],2
      * [q,t,y,x],2
      * [q,t,y,x,z],2
      * [q,t,y,z],2
      * [q,t,x],2
      * [q,t,x,z],2
      * [q,t,z],2
      * [q,x],2
      * [q,x,z],2
      * [q,z],2
      */
  }
}
