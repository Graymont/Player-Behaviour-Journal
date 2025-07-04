# Abstrak
<p style="text-align: justify;">
Minecraft adalah permainan sandbox yang memberikan kebebasan bagi pemain untuk melakukan 
berbagai aktivitas seperti menambang, membangun, dan berburu. Setiap pemain menunjukkan pola 
perilaku yang berbeda-beda. Penelitian ini bertujuan untuk mengklasifikasikan perilaku pemain 
Minecraft berdasarkan data aktivitas in-game yang dikumpulkan menggunakan plugin Spigot 
khusus bernama PlayerBehaviour. Data yang dicatat meliputi jumlah blok dihancurkan, blok 
dipasang, mob dikalahkan, item dikraft, jarak tempuh, dan berbagai aktivitas lainnya. Data 
kemudian dilabeli secara otomatis berdasarkan dominasi aktivitas tertentu, dan digunakan untuk 
melatih model klasifikasi dengan algoritma Decision Tree dan Random Forest. Hasil pelatihan 
menunjukkan bahwa algoritma Random Forest memberikan akurasi terbaik sebesar 74%, 
mengungguli Decision Tree yang hanya mencapai 39%. Model ini berpotensi digunakan untuk 
personalisasi pengalaman bermain serta mendeteksi perilaku tidak wajar pemain. 
</p>

# Content List
- Dataset hasil dari auto generate di plugin PlayerBehaviour
- Script Python untuk melakukan testing data berdasarkan model

# Python Library Installation

*Pandas*
```pip install pandas```

*Scikit-learn*
```pip install scikit-learn```

*Matplotlib*
```pip install matplotlib```

*Seaborn*
```pip install seaborn```

*Joblib*
```pip install joblib```

# PlayerBehaviour Spigot Plugin Source Code
repository link: https://github.com/Graymont/Player-Behaviour
