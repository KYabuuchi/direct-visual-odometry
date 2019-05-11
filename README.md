# Abstract
* semi-dense-visual-odometryの再現実装
# ASAP
* オンラインモード
    * Trackingの映像出力を間引く
    * Ageは高々10で頭打ち
    * Tracking終了判定，KeyFrame挿入判定，深度推定分散
    * Mapping高速化

# TODO 
* Trackingの終了判定が微妙
* 他のデータを撮影
* NewFrameの挿入条件の再検討
* 後半の方で深度推定値が変になるのは，後半の方がカメラの動きが過激だからかもしれない
* 可能なら，MappingとTrackingの並列化
* エピポーラ直線が外側にいくほど，斜めになる現象