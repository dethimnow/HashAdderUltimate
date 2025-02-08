
*********************
This is an update HashAdder script. Changes include:
*********************

Collision–List Generation:
The script computes a “twos” list from a starting point (based on the user’s public key and input parameter) and then uses parallel processing (via a ProcessPoolExecutor) to compute collision entries. The results are bucketed (using a prefix derived from the candidate’s x–coordinate) and presorted before being saved.

Multiplication Grid:
A precomputed grid is built so that the “multiply_num()” function quickly computes the elliptic curve point corresponding to a given candidate integer.

Search Routines:
Three search methods are implemented:

Lookup Grid Search / Reverse Grid Search: These iterate over “grid rows.”
Random Search: This performs a random walk through candidate keys.
easyCount() Search (New):
This adaptive iterative search uses the “density” of collision entries in the candidate’s bucket to adjust the search step size. If a candidate’s bucket is empty the step is increased; if the bucket is dense the step is reduced—narrowing the search region. Progress (iterations per second) is printed periodically.
Multi–Key Input and Device Selection:
Users may input multiple public keys (which are added if they share the same curve) and select between CPU and GPU (currently GPU falls back to CPU).


NOTICE:  YOU MUST KNOW HOW TO RUN A PYTHON PROGRAM FROM THE COMMAND LINE / TERMINAL IN ORDER TO USE THIS TOOL!!!

************
**Installation of Prerequisites**
************
Version 1.6 is the Latest Working Release and works on both Linux and Windows operating systems.
To Install the prerequisites using Linux:

Version 1.6 must be run as Administrator / "Root" if using Linux Terminal.
You must also install (as root user) "tqdm" and "tinyec" using this command:

sudo python3 -m pip install tinyec tqdm

Then run hashAdder.py as root like this:

sudo python3 hashAdder.py

To Install the prerequisites using Windows:
Use the same commands as above from a command prompt (As Administartor) without "sudo" at the begining of each command.

If using Windows you can run as administrator but it is not needed if you give file permissions to the current user.
You may need to run the most recent version from command line / terminal as an administrator if using Linux or Mac in order to create the collisionList files...

************
**OverView**
************

NOTICE:You must run the newest version from its own folder / directory. Failure to do so will result in functionality conflicts between files and will cause the program to not work properly!!!

HashAdder is a python program that allows for the searching of a single key to become searching for many different keys which all correspond mathematically to the single key, which is input by the user.
The program uses multiplication, division and exponents to create a large "collision list" of keys which drastically reduces the "worst case" time complexity of recovering a private Key.

If you would rather watch a video explaining this program, please visit my YouTube channel here:
https://www.youtube.com/@quitethecontrary1846
Video explaining hashAdder (part One now available on YouTube)

This description assumes you have a basic understanding of the math behind bitcoin being modular arithmetic/ "clock math"(Modulo - the remained after division - by some large prime number).
This is done in python using "%" operator. Example: (10*2)%13=7.
Note: Public Keys do not need to use the "%" symbol when doing arithmetic because the modular division is done automatically with in ECDSA itself. More explained about this in the HashAdder Explanation Video.

The math hashAdder uses  is similar to how "deterministic keys" are created from a master public key using random multiples.
Also explained in the video is the fact that the total search space is NOT "n-1" even though this is the largest possible private key.
Please watch video for a detailed explanation and visual proof.

The basic advantage this program has over a password cracker is the fact that with a password cracker; you are only looking for a single
password key to match out of all of the many possibilities. Because the modular arithmetic works the same with both private keys 
as it does for public keys, we can use multiplication and division to search for multiples and divisors of a public key. (the same math can
be checked using private keys modulo "n"  for accuracy). Use division to reverse the multiples and multiplication to reverse the divisors.
Note: It is only currently known how to reverse multiples and divisors of combinations of 2 and 3. HashAdder uses these multiples for its collision list.

*********************************************************************
**"Public Keys", "Private Keys", and the difference between the Two**
*********************************************************************

The math of ECDSA is done Modulo "n" where "n" = 115792089237316195423570985008687907852837564279074904382605163141518161494337 and is the largest possible bitcoin Private Key + 1.

All public keys are generated by moving a point along a plane by multiplying the x and y coordinates of that point,
(x = 55066263022277343669578718895168534326250603453777594175500187360389116729240, y = 32670510020758816978083085130507043184471273380659243275938904335757337482424) in the case of all bitcoin keys,
by a private key(integer/scalar).
This private key can be any integer value from 1 - 115792089237316195423570985008687907852837564279074904382605163141518161494336.
The resulting output of the multiplication function is a Public Key in the format of its x and y coordinates.

*******************
**Collision Lists**
*******************

HashAdder uses a faster approach to each multiplication. A faster calculator called the "multiplyNum" function provides a reduction each initial 256 bit random number(worst case) multiplication from 255 additions(for the origional calculator) maximum down to 15 additions(for hashAdders calculator). This allows for a 24 X speed up in "randomSsearch()" and "collisionListSort()" functions respectively.
 
Seperatley, when using "random Search", a key is generated, then an addition loop (via adding 1) is used to search around the randomly generated private key up and down by 100,000 by first subtracting 100,000 from the position and then adding one, 200,000 times. This gives an abstract "pie slice" effect to each search.

Considering the math is done within a modular form over as finite field...this allows for continual divisions and multiplications without ever going higher than the largest possible private key, as the counter just starts over at 1 once it reaches that point just like a clock. There are also NO negative integers...only subtraction from "n". This gives an infinite number of possible multiples and divisors.
The larger the list, the more likely you will find a collision match due to the total search space being divided by the size of the list. However larger lists take much more time to both create and to search (even using binary search).
Please keep this in mind when creating collision lists.

**********************
**System Limitations**
**********************

In the Current version, Collision Lists Creation is limited to the amount of RAM available on the device. 1GB of RAM will hold 2,881,200 keys with their identifiers.
An input of 3200 for the size of the list when input by the user will give a collision list containing 40,972,801 and will take up 20GB of RAM when creating the list and only 10 when the list is created and being used durinng search.
Sorting of the list requires double the amount of RAM while the list is being created.

*****************************
**The Math Behind hashAdder**
*****************************

It is well known that division can be done using multiplication via using decimal. For example: Some number / 2 = Some number * 0.5.

In order to divide by 2, the half position of ((n-((n-1)//2))%n)) is used where:
half = 57896044618658097711785492504343953926418782139537452191302581570759080747169

In Order to divide by 3, the third position of ((n-(n//3))%n) is used where:
third = 77194726158210796949047323339125271901891709519383269588403442094345440996225

In order to reverse the iteration of the division by 2, we multiply by the corresponding power of 2...same for 3.
Multiplication and division iterations can all be confirmed using private keys and public Keys alike by dividing a number of times and then multiplying the same number of times using the reciprocal to get back to the starting position. (As Explained in Video)
Example:
(Public Key * 4) / (4) = original Public Key

(Divide by 4 is done using (Public Key *4) * (half ^ 2)

or using the original generator as "1",

Public Key *(2^32)%n="Multiple to be Found"
(random Key Integer) *(Original Generator) = Matching Public Key to "Multiple to be Found"
"Found Key" *(half^32)%n= Private Key Recovered From Public Key

*********************************************************************
**Why is hashAdder Better Than a Typical Password Cracking Program?**
*********************************************************************

Instead of only looking for a Single "unknown" Private Key using its known public Key; we can look for billions of keys that are all different and all mathematically correspond an original key(Public Key).
If the Random Number Generator generates a private key that becomes a Public Key that matches any matching multiple or divisors of the key we are looking for, then it automatically iterates back to the marked number of positions to recover the unknown private key.
This reduces the total search space from (half of 'n-1') to ('half of n-1' // "total keys contained in the collision list").

The multiples and divisors are saved to memory in a sorted list by their first 6 prefix digits of the x-coordinate along with how many times the entered public key was multiplied or divided for get to that position in this format: (prefix Digits,("x-coordinate",(twos, threes))). this allows for exact recovery of the private key that corresponds to the public key entered at the start of the program.
This also allows for binary search to be used for each iterative try.

Tuples are used as lookup dictionaries to allow indexing, which gives constant lookup time to each iteration of binary search.

***********************
**Running the Program**
***********************

The current version requires the user have the "tinyEC.py" package installed on the system or within the same working directory as hashAdder's files using "pip".
 
When the program is run, it will ask for a number to be entered for the size of the collision list you would like to create. The number entered is doubled, 1 is added and then that total is squared, (((2*input amount)+1)**2), in order to give a total amount of multiples contained in the list.

The program will give all multiples and divisions by every combination of 2 and 3 up to double the amount the user inputs.
Example: 
if input = 3 
this will output 49 total keys in the final collision list. (((2*3)+1)**2)=49
the key is divided by 2 three times, these divisions by 2 are divided and multiplied by 3 three times
then  the key is multiplied by 2 three times and divided and multiplied by 3 three times.
giving a resulting chart like this:

****************************************************************************************************
--((key/8) x 27)--((key/4) x 27)--((key/2) x 27)--(key x 27)---(key x 54)--(key x 108)--(key x 216)-
****************************************************************************************************
--((key/8) x 9)---((key/4) x 9)----((key/2) x 9)--(key x 9)---(key x 18)---(key x 36)---(key x 72)--
****************************************************************************************************
--((key/8) x 3)---((key/4) x 3)----((key/2) x 3)--(key x 3)---(key x 6)-----(key x 12)--(key x 24)--
****************************************************************************************************
--(key/8)------------(key/4)-------(key/2)-------(key x 1)-----(key x 2)----(key x 4)---(key x 8)---
****************************************************************************************************
--((key/8)/3)------((key/4)/3)---((key/2)/3)----(key/3)----((key*2)/3)---((key*4)/3)----((key*8)/3)-
****************************************************************************************************
--((key/8)/9)-----((key/4)/9)----((key/2)/9)----(key/9)----((key*2)/9)---((key*4)/9)----((key*8)/9)-
****************************************************************************************************
-((key/8)/27)---((key/4)/27)---((key/2)/27)---(key/27)---((key*2)/27)---((key*4)/27)---((key*8)/27)-
****************************************************************************************************

