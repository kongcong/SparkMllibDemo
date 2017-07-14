package com.xj.ml.recommender

import breeze.numerics.sqrt
import org.apache.spark.rdd.RDD

/**
  * author : kongcong  
  * number : 27
  * date : 2017/7/13
  */

/**
  * 用户评分
  * @param userid 用户
  * @param itemid 评分物品
  * @param pref 评分
  */
case class ItemPref(
  val userid: String,
  val itemid: String,
  val pref: Double) extends Serializable

/**
  * 用户推荐
  * @param userid 用户
  * @param itemid 推荐物品
  * @param pref 评分
  */
case class UserRecomm(
  val userid: String,
  val itemid: String,
  val pref: Double) extends Serializable

/**
  * 相似度
  * @param itemid1 物品
  * @param itemid2 物品
  * @param similar 相似度
  */
case class ItemSimi(
  val itemid1: String,
  val itemid2: String,
  val similar: Double) extends Serializable

/**
  * 相似度计算
  * 支持：同现相似度、欧式距离相似度、余弦相似度
  */
class ItemSimilarity extends Serializable {

  def Similarity(user_rdd: RDD[ItemPref], stype: String): (RDD[ItemSimi]) = {
    val simil_rdd: RDD[ItemSimi] = stype match {
      case "cooccurrence" =>
        ItemSimilarity.CooccurrenceSimilarity(user_rdd)
    }
    simil_rdd
  }

  object ItemSimilarity {

    /**
      * 同现相似度矩阵计算
      * w(i,j) = N(i) ∩ N(j) / sqrt(N(i) * N(j))
      * @param user_rdd 用户评分
      * @return RDD[ItemSimi] 返回物品相似度
      */
    def CooccurrenceSimilarity(user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {
      // 准备数据
      val user_rdd1: RDD[(String, String, Double)] = user_rdd.map(f => (f.userid, f.itemid, f.pref))
      val user_rdd2: RDD[(String, String)] = user_rdd1.map(f => (f._1, f._2))
      // (用户，物品)笛卡尔积操作 => 物品与物品的组合
      val user_rdd3: RDD[(String, (String, String))] = user_rdd2.join(user_rdd2)
      val user_rdd4: RDD[((String, String), Int)] = user_rdd3.map(f => (f._2, 1))
      // (物品，物品，频次)
      val user_rdd5: RDD[((String, String), Int)] = user_rdd4.reduceByKey((x, y) => x + y)
      // 对角矩阵
      val user_rdd6: RDD[((String, String), Int)] = user_rdd5.filter(f => f._1._1 == f._1._2)
      // 非对角矩阵
      val user_rdd7: RDD[((String, String), Int)] = user_rdd5.filter(f => f._1._1 != f._1._2)
      // 计算同现相似度（物品1，物品2，同现频次）
      val user_rdd8: RDD[(String, ((String, String, Int), Int))] =
        user_rdd7.map(f => (f._1._1, (f._1._1, f._1._2, f._2)))
          .join(user_rdd6.map(f => (f._1._1, f._2)))
      val user_rdd9: RDD[(String, (String, String, Int, Int))] =
        user_rdd8.map(f => (f._2._1._2, (f._2._1._1, f._2._1._2, f._2._1._3, f._2._2)))
      val user_rdd10: RDD[(String, ((String, String, Int, Int), Int))] =
        user_rdd9.join(user_rdd6.map(f => (f._1._1, f._2)))
      val user_rdd11 = user_rdd10.map(f => (f._2._1._1, f._2._1._2, f._2._1._3, f._2._1._4, f._2._2))
      val user_rdd12 = user_rdd11.map(f => (f._1, f._2, (f._3 / sqrt(f._4 * f._5))))

      // 结果返回
      user_rdd12.map(f => ItemSimi(f._1, f._2, f._3))
    }

    def CosineSimilarity (user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {
      // 准备数据
      val user_rdd1: RDD[(String, String, Double)] = user_rdd.map(f => (f.userid, f.itemid, f.pref))
      val user_rdd2: RDD[(String, (String, Double))] = user_rdd1.map(f => (f._1, (f._2, f._3)))
      // (用户，物品，评分)笛卡尔积操作 => (物品1，物品2，评分1，评分2)组合
      val user_rdd3: RDD[(String, ((String, Double), (String, Double)))] = user_rdd2.join(user_rdd2)
      val user_rdd4: RDD[((String, String), (Double, Double))] =
        user_rdd3.map(f => ((f._2._1._1, f._2._2._1), (f._2._1._2, f._2._2._2)))
      // (物品1，物品2，评分1，评分2)组合 => (物品1，物品2，评分1 * 评分2)组合并累加
      val user_rdd5: RDD[((String, String), Double)] = user_rdd4.map(f => (f._1, f._2._1 * f._2._2)).reduceByKey(_ + _)
      // 对角矩阵
      val user_rdd6: RDD[((String, String), Double)] = user_rdd5.filter(f => f._1._1 == f._1._2)
      // 非对角矩阵
      val user_rdd7: RDD[((String, String), Double)] = user_rdd5.filter(f => f._1._1 != f._1._2)
      // 计算相似度
      val user_rdd8: RDD[(String, ((String, String, Double), Double))] =
        user_rdd7.map(f => (f._1._1, (f._1._1, f._1._2, f._2)))
          .join(user_rdd6.map(f => (f._1._1, f._2)))
      val user_rdd9: RDD[(String, (String, String, Double, Double))] = user_rdd8.map(f => (f._2._1._2, (f._2._1._1, f._2._1._2, f._2._1._3, f._2._2)))
      val user_rdd10: RDD[(String, ((String, String, Double, Double), Double))] = user_rdd9
          .join(user_rdd6.map(f => (f._1._1, f._2)))
      val user_rdd11 = user_rdd10.map(f => (f._2._1._1, f._2._1._2, f._2._1._3, f._2._1._4, f._2._2))
      val user_rdd12 = user_rdd11.map(f => (f._1, f._2, (f._3 / sqrt(f._4 * f._5))))
      // 结果返回
      user_rdd12.map(f => ItemSimi(f._1, f._2, f._3))
    }

    def EuclideanDistanceSimilarity(user_rdd: RDD[ItemPref]): (RDD[ItemSimi]) = {
      // 准备数据
      val user_rdd1 = user_rdd.map(f => (f.userid, f.itemid, f.pref))
      val user_rdd2 = user_rdd1.map(f => (f._1, (f._2, f._3)))
      // （用户，物品，评分）笛卡尔积操作 => （物品1，物品2，评分1，评分2）组合
      val user_rdd3 = user_rdd2.join(user_rdd2)
      val user_rdd4: RDD[((String, String), (Double, Double))] = user_rdd3
        .map(f => ((f._2._1._1, f._2._2._1), (f._2._1._2, f._2._2._2)))
      // （物品1，物品2，评分1，评分2）组合 => （物品1，物品2，评分1 - 评分2）组合并累加
      val user_rdd5: RDD[((String, String), Double)] = user_rdd4.map(f => (f._1, (f._2._1 - f._2._2) * (f._2._1 - f._2._2))).reduceByKey(_+_)
      // （物品1，物品2，评分1，评分2）组合 => （物品1，物品2,1）组合计算物品1与物品2的重叠数
      val user_rdd6 = user_rdd4.map(f => (f._1, 1)).reduceByKey(_+_)
      // 非对角矩阵
      val user_rdd7 = user_rdd5.filter(f => f._1._1 != f._1._2)
      // 计算相似度
      val user_rdd8 = user_rdd7.join(user_rdd6)
      val user_rdd9 = user_rdd8.map(f => (f._1._1, f._1._2, f._2._2 / (1 + sqrt(f._2._1))))
      // 结果返回
      user_rdd9.map(f => ItemSimi(f._1, f._2, f._3))
    }
  }

}
