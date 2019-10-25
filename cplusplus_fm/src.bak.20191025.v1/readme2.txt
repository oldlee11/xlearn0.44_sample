src->solver->train_main.cc

run->edit-config..->Allaplication->
programe arg.. =
./split.train.txt -v ./split.test.txt -s 1 -e 200 -nthread 8 -k 100 -x acc -p sgd -r 0.1
./split.train.txt -v ./split.test.txt -s 1 -e 200 -nthread 8 -k 100 -x acc -p sgd -r 0.1 -pre split.train.txt.model.txt

working dir.. =
/data/ming.li/workspace/TrainModel/api/fm/cplusplus_fm/demo/classification/criteo_ctr





========
/data/ming.li/workspace/TrainModel/api/fm/cplusplus_fm/demo/classification/kuwo
scp ming.li@10.0.34.12:/data/ming.li/workspace/TrainModel/userdb/userPlayMusicSeq/DataSet/20190916/cplusplus/ctr_dateset.txt ./
scp ming.li@10.0.34.12:/data/ming.li/workspace/TrainModel/userdb/userPlayMusicSeq/DataSet/20190916/cplusplus/init.modle.txt ./

working dir.. =
/data/ming.li/workspace/TrainModel/api/fm/cplusplus_fm/demo/classification/kuwo

run->edit-config..->Allaplication->
# v1 -----------
programe arg.. =
ctr_dateset.txt.train -v ctr_dateset.txt.test -s 1 -e 200 -nthread 16 -k 100 -x acc -p sgd -r 0.1 -pre init.modle.txt
[------------] Epoch      Train log_loss       Test log_loss       Test Accuarcy     Time cost (sec)
[    0%      ]     1            1.061088            0.815298            0.544966              593.03
[    1%      ]     2            0.752121            0.765455            0.564894              625.98
[    1%      ]     3            0.705528            0.749346            0.561206              593.71
[    2%      ]     4            0.681982            0.764764            0.514351              621.28
[    2%      ]     5            0.666433            0.743552            0.540279              605.94


# v4 -----------
programe arg.. =
ctr_dateset.txt.v4.train -v ctr_dateset.txt.v4.test -s 1 -e 200 -sw 200 -nthread 16 -k 100 -x acc -p sgd -r 0.1 -pre init.modle.txt
[------------] Epoch      Train log_loss       Test log_loss       Test Accuarcy     Time cost (sec)
[    0%      ]     1            0.418037            0.368547            0.838985               22.17
[    1%      ]     2            0.319343            0.337828            0.856801               21.32
[    1%      ]     3            0.276290            0.322408            0.867844               21.70
[    2%      ]     4            0.245823            0.310691            0.870687               19.05
[    2%      ]     5            0.222077            0.301518            0.876752               21.78
[    3%      ]     6            0.202675            0.297215            0.880435               21.44
[    3%      ]     7            0.186313            0.292034            0.882746               20.35
[    4%      ]     8            0.172186            0.289969            0.880362               20.94
[    4%      ]     9            0.159854            0.284806            0.884130               20.90
[    5%      ]    10            0.149007            0.282068            0.886326               20.98
[    5%      ]    11            0.139330            0.281019            0.886589               21.17
[    6%      ]    12            0.130644            0.281369            0.885746               21.33
[    6%      ]    13            0.122831            0.279955            0.889141               20.83
[    7%      ]    14            0.115783            0.279575            0.888572               20.63
[    7%      ]    15            0.109349            0.278909            0.889662               21.22
[    8%      ]    16            0.103463            0.280081            0.889215               21.45
[    8%      ]    17            0.098057            0.279820            0.889886               21.08
[    9%      ]    18            0.093085            0.281133            0.890518               21.37
[    9%      ]    19            0.088521            0.281172            0.890165               21.32
[   10%      ]    20            0.084272            0.282992            0.890071               21.13
[   10%      ]    21            0.080337            0.283403            0.890222               20.65
[   11%      ]    22            0.076702            0.284704            0.890126               21.14
[   11%      ]    23            0.073306            0.286441            0.889968               19.85
[   12%      ]    24            0.070140            0.287211            0.890067               20.18
[   12%      ]    25            0.067196            0.290467            0.890581               21.63
[   13%      ]    26            0.064412            0.291277            0.890448               21.83
[   13%      ]    27            0.061840            0.292061            0.890149               20.75
[   14%      ]    28            0.059443            0.293448            0.890023               21.28



# v2 -----------
programe arg.. =
ctr_dateset.txt.v2.train -v ctr_dateset.txt.v2.test -s 1 -e 200 -nthread 16 -k 100 -x acc -p sgd -r 0.1 -pre init.modle.txt
[------------] Epoch      Train log_loss       Test log_loss       Test Accuarcy     Time cost (sec)
[    0%      ]     1            0.891198            0.774287            0.555021               26.52
[    1%      ]     2            0.721421            0.752007            0.569814               26.10
[    1%      ]     3            0.681383            0.736198            0.555529               26.17
[    2%      ]     4            0.657203            0.731957            0.564498               25.42
[    2%      ]     5            0.638779            0.731179            0.556199               25.73
[    3%      ]     6            0.622990            0.740560            0.546851               25.53
[    3%      ]     7            0.608457            0.739527            0.554040               25.50
[    4%      ]     8            0.594564            0.757777            0.556776               25.47
[    4%      ]     9            0.581025            0.764111            0.536569               25.64
[    5%      ]    10            0.567596            0.766634            0.549980               25.36
[    5%      ]    11            0.554562            0.777746            0.548794               25.74
[    6%      ]    12            0.541542            0.778975            0.539269               25.71
[    6%      ]    13            0.528691            0.794660            0.543751               24.81
[    7%      ]    14            0.516104            0.800049            0.537948               25.44
[    7%      ]    15            0.503540            0.824165            0.543573               25.46
[    8%      ]    16            0.491544            0.835648            0.541950               25.78
[    8%      ]    17            0.479673            0.846811            0.540519               25.79
[    9%      ]    18            0.468437            0.847657            0.535025               25.93







