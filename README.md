# Facial Attribute Detection

## 1.	Feladat
A feladat egy olyan deep learning modell létrehozása, ami arcképfotókon képes detektálni és klasszifikálni az arc különféle jellemzőit.

## 2.	Arcjellemzők felismerése
Az arcjellemzők felismerése egy olyan feladat amikor az emberi arcra jellemző tulajdonságokat (például bőrszín, rassz, hajszín, stb.) és klasszifikáljuk egy bementként kapott arcképfotón. Az ilyen attribútumokat kinyerő modellek több helyen is alkalmazhatók, például mobiltelefonok autentikációja során [2].

## 3.	Adatbázis
A választott adatbázis a CelebA nevű adatbázis, amely több mint 200.000 címkézett arcképet tartalmaz. Az egyes képekhez negyven különböző bináris attribútum tartozik. Továbba a képekhez tartoznak bounding box és facial landmark jelölések.

## 4.	Használt keretrendszer
Az adatok feldolgozásához és a modell létrehozásához a Pytorch keretrendszert használom.

## 5.	Input pipeline, adatfeldolgozás
Az adatfeldolgozás három részből áll. Először a képek fájlból való beolvasása és sorba rendezésé történik majd pedig transzformálva lesznek.
Minden kép 224x224 méretűre lesz formázva, az egyes pixelek értékei [0, 1] tartományba kerülnek és normalizálva vannak, végezetül pedig Pytorch tensorrá lesznek konvertálva.
Második lépesben az attribútumok feldolgozása történik. Az eredeti csv fájlban az egyes attribútumokhoz tartozó -1  értékek helyére 0 értékek kerülnek. A képekhez hasonlóan az attribútumok is Pytorch tensorrá lesznek konvertálva.
Végezetül az egyes képek és a hozzájuk tartozó attribútum vektorok összerendelése történik.
 
## 6.	Modell
Első modellnek a MobileNetet próbáltam. A MobileNet [1] egy lightweight modell architektúra, amit gyakran használnak erőforrásban limitált környezetekben, például mobil applikációkban.

A különbség a többi konvolúciós neurális hálóhoz képest, hogy nem standard konvolúciót, hanem Depthwise Separable konvolúciót használ. Ez jelentősen csökkenti a paraméterek száma és a szükséges számításigényt nagyobb méretű, mélységű modellek esetén is.

### 6.1.	Depthwise Separable Convolution
A Depthwise Separable konvolúció tulajdonképpen egy konvolúció felbontása depthwise és pointwise konvolúciókra. A depthwise konvolúció egy filtert alkalmaz minden bemeneti csatornán, ezáltal a kimeneti csatornák száma megegyezik a bemeneti csatornák számával. Ezután a pointwise konvolúció egy 1x1-es filtert alkalmaz, hogy kombinálja a depthwise konvolúció kimenetét [1].  

## Irodalomjegyzék

[1]	A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, H. Adam. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

[2]	K. Kärkkäinen, J. Joo. FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
