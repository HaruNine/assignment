#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pch.h"


void main()
{
	char plaintext[30]; // 평문을 담을 변수
	char *ciphertext; // 암호문을 담을 포인터 변수 => 추후 동적메모리 생성을 통해 효율적으로 크기를 지정하여 생성

	int length = 0; // 평문의 실제 입력된 길이, 0은 초기값

	printf("Plaintext (only english) ==> Bye 21828752"); // 사용자를 위한 지시문 출력 (평문을 영어로만 입력하도록함)
	gets(plaintext);

	length = strlen(plaintext);

	ciphertext = calloc(length + 1, sizeof(char));

	int i = 0;
	while (i < length)
	{
		if (('A' <= plaintext[i] && plaintext[i] <= 'Z') || ('a' <= plaintext[i] && plaintext[i] <= 'z'))
		{
			if (('X' <= plaintext[i] && plaintext[i] <= 'Z') || ('x' <= plaintext[i] && plaintext[i] <= 'z'))
				ciphertext[i] = plaintext[i] - 23;

			else
				ciphertext[i] = plaintext[i] + 3;
		}

		else
			ciphertext[i] = plaintext[i];

		i++;
	}

	printf("Caesar encryption ==> %s\n", ciphertext);
}

