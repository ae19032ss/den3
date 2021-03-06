/*************************************************/
/*          ハードウェアチェックプログラム　　　 */
/*  　　　　　      　2002.8.18     by AND      */
/*************************************************/
#include <pic.h>
__CONFIG(0xFFFA); /* 初期設定 CP:OFF,PT:OFF,WT:OFF,HS */
#define T_MAX 30 /* 300msec周期でモータをON,OFFする */
#define COUNT 3 /* 繰り返し回数 */

wait0(short k)
{
/* 約(k×0.01)秒 のウェイト*/
	short i;
	short j; /* 16 Bit 変数の宣言 */
   	for(j=0;j<k;j++){ /* (k×3000)回の繰り返し */
		for(i=0;i<3000;i++){ /* 時間待ち */
		}
	}
}

main(void)
{
	short i,j;  /* 16ビット整数型変数i,jの宣言 */
	unsigned char x;  /* 符号なし8ビット整数型変数xの宣言 */
	x=0x01;  /* xに2進数の00000001を代入 */

	TRISA = 0xFC; /* A 0,1:output, 2,3,4:input */
	TRISB = 0xC7; /* B0,1,2:input, B3,4,5:LEDoutput, other bits input */

	PORTA=0x00; /* モータ OFF */
	for(i=0;i<3;i++){
		PORTB=0x08; /* LED左点灯 */
		wait0(50); /* 0.5秒ウェイト */
		PORTB=0x10;    /* LED中点灯 */
		wait0(50); /* 0.5秒ウェイト */
		PORTB=0x20;    /* LED右点灯 */
		wait0(50); /* 0.5秒ウェイト */
	}

	PORTB=0x00; /* LED全消灯 */

	while(1){/* 無限ループ */
		RB3=RB0; /* センサ入力→LED表示 */
		RB4=RB1; /* センサ入力→LED表示 */
		RB5=RB2; /* センサ入力→LED表示 */
		wait0(3); /* 約30msecウェイト */

		if(RA4==0) /*もし、スイッチが押されたら*/
		{
			PORTA=0x01; /*右車輪回転*/
			wait0(150); /*1.5sec待ち*/
			PORTA=0x00; /*全車輪ストップ*/
			PORTA=0x02; /*左車輪回転*/
			wait0(150); /*1.5sec待ち*/
			PORTA=0x03; /*両車輪回転*/
			wait0(150); /*1.5sec待ち*/
		}
		PORTA=0x00;         /*全車輪ストップ*/


	}
}
	