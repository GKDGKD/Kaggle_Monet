#!/bin/bash
echo "hello world!"
your_name="Donald Trump"
echo "your name is ${your_name}"
for i in a b c d
do
	echo "string: ${i}"
done

str='this is a string.'
echo $str
echo "len(str) = ${#str}"
echo "len(str[0]) = ${#str[0]}"
echo "len(str[1]) = ${#str[1]}"
echo "str[0:4] = ${str:1:4}. Hello, my name is ${your_name}! I wanna go up to the sun!"

a=20
b=10
add=`expr $a + $b`
mul=`expr $a \* $b`
echo "a = $a, b = $b."
echo "a + b = $add"
echo "a * b = $mul"
