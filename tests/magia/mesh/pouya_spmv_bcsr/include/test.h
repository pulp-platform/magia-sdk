#ifndef TEST_H
#define TEST_H

#include <stdint.h>

/*
A: 64x64
x: 64x1
y: 64x1

All values are IEEE FP16 encoded in uint16_t.
*/

#define M 64
#define N 64
#define MAX_NNZ (M * N)

/* FP16 constants */
#define F16_0   0x0000
#define F16_1   0x3C00
#define F16_2   0x4000
#define F16_3   0x4200
#define F16_4   0x4400
#define F16_5   0x4500
#define F16_6   0x4600
#define F16_7   0x4700
#define F16_8   0x4800
#define F16_9   0x4880
#define F16_10  0x4900
#define F16_11  0x4980
#define F16_12  0x4A00
#define F16_13  0x4A80
#define F16_14  0x4B00
#define F16_15  0x4B80
#define F16_16  0x4C00
#define F16_17  0x4C40
#define F16_18  0x4C80
#define F16_19  0x4CC0
#define F16_20  0x4D00
#define F16_21  0x4D40
#define F16_22  0x4D80
#define F16_23  0x4DC0
#define F16_24  0x4E00
#define F16_25  0x4E40
#define F16_26  0x4E80
#define F16_27  0x4EC0
#define F16_28  0x4F00
#define F16_29  0x4F40
#define F16_30  0x4F80
#define F16_31  0x4FC0
#define F16_32  0x5000
#define F16_33  0x5020
#define F16_34  0x5040
#define F16_35  0x5060
#define F16_36  0x5080
#define F16_37  0x50A0
#define F16_38  0x50C0
#define F16_39  0x50E0
#define F16_40  0x5100
#define F16_41  0x5120
#define F16_42  0x5140
#define F16_43  0x5160
#define F16_44  0x5180
#define F16_45  0x51A0
#define F16_46  0x51C0
#define F16_47  0x51E0
#define F16_48  0x5200
#define F16_49  0x5210
#define F16_50  0x5220
#define F16_51  0x5230
#define F16_52  0x5240
#define F16_53  0x5250
#define F16_54  0x5260
#define F16_55  0x5270
#define F16_56  0x5280
#define F16_57  0x5290
#define F16_58  0x52A0
#define F16_59  0x52B0
#define F16_60  0x52C0
#define F16_61  0x52D0
#define F16_62  0x52E0
#define F16_63  0x52F0
#define F16_64  0x5300


#define F16_84    0x5540
#define F16_96    0x5600
#define F16_108   0x56C0
#define F16_132   0x5820
#define F16_144   0x5880
#define F16_156   0x58E0
#define F16_180   0x59A0
#define F16_192   0x5A00
#define F16_204   0x5A60
#define F16_228   0x5B20
#define F16_240   0x5B80
#define F16_252   0x5BE0
#define F16_276   0x5C50
#define F16_288   0x5C80
#define F16_300   0x5CB0
#define F16_324   0x5D10
#define F16_336   0x5D40
#define F16_348   0x5D70
#define F16_372   0x5DD0
#define F16_384   0x5E00
#define F16_396   0x5E30
#define F16_420   0x5E90
#define F16_432   0x5EC0
#define F16_444   0x5EF0
#define F16_468   0x5F50
#define F16_480   0x5F80
#define F16_492   0x5FB0
#define F16_516   0x6008
#define F16_528   0x6020
#define F16_540   0x6038
#define F16_564   0x6068
#define F16_576   0x6080
#define F16_588   0x6098
#define F16_612   0x60C8
#define F16_624   0x60E0
#define F16_636   0x60F8
#define F16_660   0x6128
#define F16_672   0x6140
#define F16_684   0x6158
#define F16_708   0x6188
#define F16_720   0x61A0
#define F16_732   0x61B8
#define F16_756   0x61E8
#define F16_768   0x6200
#define F16_816   0x6260
#define F16_864   0x62C0
#define F16_912   0x6320
#define F16_960   0x6380
#define F16_1008  0x63E0
#define F16_1056  0x6420
#define F16_1104  0x6450
#define F16_1152  0x6480
#define F16_1200  0x64B0
#define F16_1248  0x64E0
#define F16_1296  0x6510
#define F16_1344  0x6540
#define F16_1392  0x6570
#define F16_1440  0x65A0
#define F16_1488  0x65D0
#define F16_1536  0x6600

#define FP16(i) F16_##i

/*
Dense matrix
*/
static uint16_t A_dense[M * N] = {
#define R(i) \
F16_0,F16_0,F16_0,FP16(i),F16_0,F16_0,F16_0,F16_0,\
F16_0,FP16(i),F16_0,F16_0,F16_0,F16_0,F16_0,FP16(i),\
F16_0,FP16(i),F16_0,F16_0,F16_0,FP16(i),F16_0,F16_0,\
F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,\
F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,\
F16_0,F16_0,F16_0,F16_0,F16_0,FP16(i),F16_0,FP16(i),\
F16_0,FP16(i),F16_0,F16_0,F16_0,FP16(i),F16_0,FP16(i),\
F16_0,F16_0,F16_0,F16_0,F16_0,FP16(i),F16_0,FP16(i)

#define S(i) \
FP16(i),F16_0,FP16(i),F16_0,F16_0,F16_0,FP16(i),F16_0,\
F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,\
FP16(i),F16_0,F16_0,F16_0,FP16(i),F16_0,F16_0,F16_0,\
FP16(i),F16_0,FP16(i),F16_0,F16_0,F16_0,F16_0,F16_0,\
FP16(i),F16_0,FP16(i),F16_0,F16_0,F16_0,FP16(i),F16_0,\
F16_0,F16_0,FP16(i),F16_0,F16_0,F16_0,F16_0,F16_0,\
FP16(i),F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,\
F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0,F16_0

S(1), R(2), S(3), R(4), S(5), R(6), S(7), R(8),
S(9), R(10), S(11), R(12), S(13), R(14), S(15), R(16),
S(17), R(18), S(19), R(20), S(21), R(22), S(23), R(24),
S(25), R(26), S(27), R(28), S(29), R(30), S(31), R(32),
S(33), R(34), S(35), R(36), S(37), R(38), S(39), R(40),
S(41), R(42), S(43), R(44), S(45), R(46), S(47), R(48),
S(49), R(50), S(51), R(52), S(53), R(54), S(55), R(56),
S(57), R(58), S(59), R(60), S(61), R(62), S(63), R(64)

#undef R
#undef S
};

/*
x = alternating 1 and 2
*/
static uint16_t x[N] = {
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,
    F16_1,F16_2,F16_1,F16_2,F16_1,F16_2,F16_1,F16_2
};

/*
output
*/
static uint16_t y[M] = {0};

/*
Expected output:
you should convert the expected values to FP16 too
(or temporarily disable checking until we generate exact FP16 hex list).
*/
static uint16_t y_expected[M] = {
    F16_12,   F16_48,   F16_36,   F16_96,   F16_60,   F16_144,  F16_84,   F16_192,
    F16_108,  F16_240,  F16_132,  F16_288,  F16_156,  F16_336,  F16_180,  F16_384,
    F16_204,  F16_432,  F16_228,  F16_480,  F16_252,  F16_528,  F16_276,  F16_576,
    F16_300,  F16_624,  F16_324,  F16_672,  F16_348,  F16_720,  F16_372,  F16_768,
    F16_396,  F16_816,  F16_420,  F16_864,  F16_444,  F16_912,  F16_468,  F16_960,
    F16_492,  F16_1008, F16_516,  F16_1056, F16_540,  F16_1104, F16_564,  F16_1152,
    F16_588,  F16_1200, F16_612,  F16_1248, F16_636,  F16_1296, F16_660,  F16_1344,
    F16_684,  F16_1392, F16_708,  F16_1440, F16_732,  F16_1488, F16_756,  F16_1536
};

#endif