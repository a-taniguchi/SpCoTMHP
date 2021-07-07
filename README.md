# SpCoTMHP
Topometric-SpCoSLAM (Hierarchical-SpCoNavi)

## Folders
 - /src/: ソースコード
   - /learning/: 場所学習用コード・結果の描画コード（albert-b用のみ）
   - /planning/: パスプランニング用コード（spconavi_rosリポジトリのコピーのため、SIGVerse用のみ）
 - /SIGVerse/: SIGVerseでの家庭環境データセット（SpCoNaviのまま）・学習結果ファイル（SpCoTMHP未適用）
   - /data/: 場所概念の学習結果
   - /dataset/: 場所概念学習用のデータセット（一部、TMHP用に修正）
   - /learning/: 3LDK datasetでの場所概念学習用コード・描画用コード
   - /planning/: SpCoNaviのコードのまま未変更
 - /albert-b/: 実環境オープンデータセット（SpCoTMHP用に変更済み）・学習結果ファイル
   - /data/: 場所概念の学習結果
   - /dataset/: 場所概念学習用のデータセット（一部、TMHP用に修正）
   - /learning/: albert-b datasetでの場所概念学習用コード・描画用コード（/src/フォルダに移植済み）
   - /planning/: SpCoNaviのコードのまま未変更

## TODO
 - /src/のフォルダを再整理中（保留）

## 現状
 -  /spconavi_ros/の中身をこのリポジトリに反映させた（SpCoNaviではなく、SpCoTMHPとして直接中身をいじるため）
 - albert-bの学習用プログラム・結果の描画プログラム完了
 - 不要なファイルの削除
 - SIGVerseの学習用プログラム・結果の描画プログラム完了(2021/07/07)
 - 3LDK_01の環境でSpCoTMHPの学習結果ファイルを作成(2021/07/07)
