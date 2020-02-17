/*************************************************/
/*          �n�[�h�E�F�A�`�F�b�N�v���O�����@�@�@ */
/*  �@�@�@�@�@      �@2002.8.18     by AND      */
/*************************************************/
#include <pic.h>
__CONFIG(0xFFFA); /* �����ݒ� CP:OFF,PT:OFF,WT:OFF,HS */
#define T_MAX 30 /* 300msec�����Ń��[�^��ON,OFF���� */
#define COUNT 3 /* �J��Ԃ��� */

wait0(short k)
{
/* ��(k�~0.01)�b �̃E�F�C�g*/
	short i;
	short j; /* 16 Bit �ϐ��̐錾 */
   	for(j=0;j<k;j++){ /* (k�~3000)��̌J��Ԃ� */
		for(i=0;i<3000;i++){ /* ���ԑ҂� */
		}
	}
}

main(void)
{
	short i,j;  /* 16�r�b�g�����^�ϐ�i,j�̐錾 */
	unsigned char x;  /* �����Ȃ�8�r�b�g�����^�ϐ�x�̐錾 */
	x=0x01;  /* x��2�i����00000001���� */

	TRISA = 0xFC; /* A 0,1:output, 2,3,4:input */
	TRISB = 0xC7; /* B0,1,2:input, B3,4,5:LEDoutput, other bits input */

	PORTA=0x00; /* ���[�^ OFF */
	for(i=0;i<3;i++){
		PORTB=0x08; /* LED���_�� */
		wait0(50); /* 0.5�b�E�F�C�g */
		PORTB=0x10;    /* LED���_�� */
		wait0(50); /* 0.5�b�E�F�C�g */
		PORTB=0x20;    /* LED�E�_�� */
		wait0(50); /* 0.5�b�E�F�C�g */
	}

	PORTB=0x00; /* LED�S���� */

	while(1){/* �������[�v */
		RB3=RB0; /* �Z���T���́�LED�\�� */
		RB4=RB1; /* �Z���T���́�LED�\�� */
		RB5=RB2; /* �Z���T���́�LED�\�� */
		wait0(3); /* ��30msec�E�F�C�g */

		if(RA4==0) /*�����A�X�C�b�`�������ꂽ��*/
		{
			PORTA=0x01; /*�E�ԗ։�]*/
			wait0(150); /*1.5sec�҂�*/
			PORTA=0x00; /*�S�ԗփX�g�b�v*/
			PORTA=0x02; /*���ԗ։�]*/
			wait0(150); /*1.5sec�҂�*/
			PORTA=0x03; /*���ԗ։�]*/
			wait0(150); /*1.5sec�҂�*/
		}
		PORTA=0x00;         /*�S�ԗփX�g�b�v*/


	}
}
	