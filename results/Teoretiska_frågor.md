# Teoretiska frågor

## 1. Hur är AI, Maskininlärning och Deep Learning relaterat?
AI är det breda fältet som går ut på att få datorer att göra smartare saker som vi människor annars bara kan, som att lösa problem eller fatta beslut. ML, är en del av AI där datorer lär sig från data på egen hand, utan att vi behöver ge dem exakta instruktioner. DL, är en avancerad typ av ML som använder neurala nätverk med många lager för att hitta och förstå komplexa mönster, till exempel att känna igen objekt i bilder eller förstå talat språk.

## 2. Hur är Tensorflow och Keras relaterat?
Tensorflow är det underliggande lagret där de tunga beräkningarna sker och är själva motorn för maskininlärning. Keras är ett lager ovanpå det som förenklar användandet av Tensorflow, som ett gränsnitt du jobbar med fast där Tensoflow är den som utför operationerna i bakgrunden.

## 3. Vad är en parameter? Vad är en hyperparameter?
En parameter är något som modellen själv räknar ut och finjusterar när den tränas på data, till exempel vikterna i ett neuralt nätverk som bestämmer hur viktig varje del av datan är. En hyperparameter är något du själv bestämmer innan träningen startar, som påverkar hur modellen lär sig, till exempel hur många lager nätverket ska ha eller hur snabbt det ska lära sig.

## 4. När man skall göra modellval och modellutvärdering kan man använda tränings-, validerings- och testdataset. Förklara hur de olika delarna kan användas.
Träningsdata är datan som modellen använder för att lära sig. Det är här den justerar sina parametrar för att hitta mönster i datan. Valideringsdata används under träningen för att testa modellen och justera hyperparametrar för att se till att modellen inte bara minns träningsdatan utan lär sig på ett bra sätt. Testdata är helt ny data som modellen får se först i slutet, för att kolla hur bra den faktiskt fungerar på data den aldrig sett förut.

## 5. Förklara vad nedanstående kod gör:

Ett neuralt nätverk skapas för att räkna ut sannolikheten vid ex. klassificiering.

`n_cols = x_train.shape[1]`
Antalet kolumner hämtas ifrån träningsdatat.

`nn_model = Sequential()`  
Här skapas en sekventiell modell

`nn_model.add(Dense(100, activation='relu', input_shape=(n_cols, )))`  
Ett lager med 100 neuroner skapas, fullt kopplade till inputlagret som har antal neuroner som det fanns kolumner i träningsdatat. ReLu används som aktiveringsfunktion.

`nn_model.add(Dropout(rate=0.2))`  
För att undvika överanpassning så stängs 20% av noderna av slumpmässigt.

`nn_model.add(Dense(50, activation="relu"))`  
Ett andra lager med 50 neuroner skapas med ReLu som aktiveringsfunktion.

`nn_model.add(Dense(1, activation="sigmoid"))`  
Ett utdatalager skapas med bara 1 neuron och sigmoid aktiveringsfunktion (tal mellan 0 och 1)

`nn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy" ])`  
Modellen kompileras med Adam optimeringsalgoritm till vikterna, binary crossentropy är förlustfunktionen och accuracy mäter hur bra modellen är på att gissa rätt.

`early_stopping_monitor = EarlyStopping(patience=5)`  
Träningen stoppas om modellen inte förbättras efter 5 epoker.

`nn_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[early_stopping_monitor])`  
Modellen tränas på träningsdata 80% och tränas maximalt i 100 epoker om den inte stängs innan dess med early stopping.

## 6. Vad är syftet med att regularisera en modell?
Syftet är att förhindra överanpassning och att modellen ska lära sig träningsdatan alltför bra och därför bli sämre på ny, osedd data sedan.

## 7. ”Dropout” är en regulariseringsteknik, vad är det för något?
Dropout stänger slumpmässigt av en andel neuroner i ett lager under träningen för att tvinga det att inte lita för mycket på specifika noder intill. Den får då istället lära sig mönster som generaliserar bättre.

## 8. ”Early stopping” är en regulariseringsteknik, vad är det för något?
Early stopping stoppar träningen av en modell om den inte förbättras efter ett visst antal epoker mot valideringsdatat. Detta förhindrar överanpassning och hjälper modellen att generalisera bättre sedan.

## 9. Din kollega frågar dig vilken typ av neuralt nätverk som är populärt för bildanalys, vad svarar du?
Jag skulle svara att CNN (convolutional neural networks) är bäst på detta. 

## 10. Förklara översiktligt hur ett ”Convolutional Neural Network” fungerar.
CNN är ett neuralt nätverk som är bra på att jobba med bilder, till exempel för att känna igen objekt, klassificera bilder eller hitta ansikten. Det börjar med konvolutionslager, där små filter "skannar" bilden för att hitta grundläggande mönster, som kanter eller texturer. Dessa filter skapar feature maps, som är kartor över viktiga detaljer i bilden. Efter det används en aktiveringsfunktion för att göra det möjligt för modellen att lära sig mer komplicerade mönster.
Sen kommer pooling-lager, som krymper storleken på feature maps för att spara datorkraft och minska risken för överanpassning. Efter flera sådana lager "plattas" resultaten ut och skickas till fully connected layers, som kombinerar allt modellen lärt sig för att till exempel säga vad som finns i bilden.
Under träningen använder CNN backpropagation, där modellens vikter justeras för att göra prediktionerna så korrekta som möjligt genom att minska felen.

## 11. Vad gör nedanstående kod?
`model.save("model_file.keras")`  
`my_model = load_model("model_file.keras")`  
Här sparas modellen "model" ner till disk. I nästa steg hämtas/öppnas den upp igen med ett nytt namn "my_model".
Det här är praktiskt för att slippa träna om modellen från början varje gång du vill använda den – du kan bara fortsätta där du slutade. Dessutom är det smidigt för att spara en backup av modellen eller dela den med andra.

## 12. Deep Learning modeller kan ta lång tid att träna, då kan GPU via t.ex. Google Colab skynda på träningen avsevärt. Skriv mycket kortfattat vad CPU och GPU är.
CPU är datorns hjärna och arbetar steg för steg. Den är bra på komplicerade uppgifter som kräver mycket logik. GPU, som från början togs fram för att hantera grafik, kan göra många enklare uppgifter parallellt, vilket gör den mycket snabbare till stora mängder beräkningar.



