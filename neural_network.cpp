#include <stdio.h>  // printf
#include <stdlib.h> //rand
#include <math.h>   // 自然対数eの指数対数expを使う

#define IN_NUM 2  // 入力層のユニット数
// #define HID_NqUM 3 // 中間層
#define HID_NUM 4 // 中間層
#define OUT_NUM 2 // 出力層 (AND回路 + OR回路の出力層)
#define PAT_NUM 4 // 学習データ数
#define ETA 0.2   // 学習率

double w1[IN_NUM+1][HID_NUM]; // 第1層の結合重み。 + 1はバイアスユニット
double w2[HID_NUM+1][OUT_NUM]; // 第2層の結合重み。
double yy1[HID_NUM]; // 中間層の出力(学習で使用)。本当はy1。y1は予約のため、yy1

/* 学習データ(入力信号) */
double input[PAT_NUM][IN_NUM] = {{0, 0},
                                 {0, 1},
                                 {1, 0},
                                 {1, 1}}; 

/* 学習データ(教師信号) */
double teach[PAT_NUM][OUT_NUM] = {{0, 0},
                                   {0, 1},
                                   {0, 1},
                                   {1, 1}}; 

// 今回の目的
// 0 0 =>0 0
// 0 1 =>0 1
// 1 0 =>0 1
// 1 1 =>1 1
// になるように学習させる

// 0以上1未満の乱数を生成する関数
double random(){
    return (double)rand() /(1.0 + RAND_MAX);
}

// シグモイド関数
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// 結合重みの初期化関数。-0.5から0.5の乱数でw1とw2を初期化する
void init(){
    int i,j;
    // w1の初期化
    for(i = 0; i < IN_NUM+1; i++){
        for(j = 0; j < HID_NUM; j++){
            w1[i][j] = random() - 0.5;
        }
    }
    // w2の初期化
    for(i = 0; i < HID_NUM+1; i++){
        for(j = 0; j < OUT_NUM; j++){
            w2[i][j] = random() - 0.5;
        }
    } 
}

// 出力関数(バイアスユニットが最後尾にある場合)
// 引数in :入力信号
// 引数out :出力信号(呼び出しもとに出力を配列で返却する)
void output(double* in, double* out){
    int i, j;
    double s; // ユニットに中に保存する変数
    //init();
    // 中間層の計算
    for (j = 0; j < HID_NUM; j++)
    {
        s = 0;
        for (i = 0; i < IN_NUM; i++)
        {
            s += w1[i][j] * in[i];
        }
        s += w1[i][j]; // バイアスユニット
        yy1[j] = sigmoid(s);
    }
    // 出力層の計算
    for (j = 0; j < OUT_NUM; j++)
    {
        s = 0;
        for (i = 0; i < HID_NUM; i++)
        {
            s += w2[i][j] * yy1[i];
        }
        s += w2[i][j]; // バイアスユニット
        out[j] = sigmoid(s);
    }
}

// 誤差逆伝播法
// 引数in: 入力信号
// 引数te: 教師信号
// 戻り値は学習誤差
double learn(double* in, double* te){
    int i, j, n;
    double out[OUT_NUM]; // 出力信号を保存おく配列
    double delta[OUT_NUM]; // δ (2)_nを保存する変数
    double error = 0; // 学習誤差

    // output関数を呼び出し、inを入力したときの出力を得る
    output(in, out);

    // 誤差の計算
    for(j = 0; j < OUT_NUM; j++){
        error += (te[j] - out[j]) * (te[j] - out[j]);
    }

    // 第2層の結合重みの学習
    for(j = 0; j < OUT_NUM; j++){
        // delta = f'*(t-y) =f*(1-f)*(t-y) = out*(1-out)*(t-out)
        delta[j] = out[j] * (1 - out[j]) *(te[j] - out[j]);
        for(i = 0; i < HID_NUM; i++){
            w2[i][j] += ETA * delta[j] * yy1[i];
        }
        // バイアスユニットの学習
        w2[i][j] += ETA * delta[j];
    }

    // 第1層の結合重みの学習
    for(j = 0; j < HID_NUM; j++){
        double del = 0;
        for(n = 0; n < OUT_NUM; n++){
            del += delta[n] * w2[j][n];
        }
    
        for(i = 0;  i < IN_NUM; i++){
            // delta = f'*(t-y) =f*(1-f)*(t-y) = out*(1-out)*(t-out)
            w1[i][j] += ETA * yy1[j] * (1 - yy1[j]) * del * in[i];  
        }
        // バイアスユニットの学習
        w1[i][j] += ETA * yy1[j] * (1 - yy1[j]) * del;
    }

    return error;
}

int main(void)
{
    int i;
    FILE *fp;
    fp = fopen("output.txt", "wt");

    if (fp == NULL) {         
        printf("cannot open\n");       
        exit(1);                  
    }

    // 学習のプログラム
    for( i = 0; i < 10000; i++){
        int n = (int)(PAT_NUM * random());
        double e = learn(input[n], teach[n]);
        printf("%d回目error=%f\n", i, e);
        fprintf(fp, "%d回目error=%f\n", i, e);
    }

    // 入出力チェック
    for(i = 0; i < PAT_NUM; i++){
        double out[OUT_NUM]; // NWの出力が入る配列
        output(input[i], out);
        printf("%f %f -> %f %f\n", input[i][0], input[i][1], out[0], out[1]);
        fprintf(fp, "%f %f -> %f %f\n", input[i][0], input[i][1], out[0], out[1]);
    }

    fclose(fp);
    return 0;
}