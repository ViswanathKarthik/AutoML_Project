import { SampleDataset } from "../types";

export const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: "iris",
    name: "Iris Flower Classifier",
    description: "The classic Fisher's Iris dataset for multi-class classification. Predict the species of iris flowers based on length and width of sepals and petals.",
    targetCol: "species",
    problemType: "classification",
    csvData: `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
4.7,3.2,1.3,0.2,Setosa
4.6,3.1,1.5,0.2,Setosa
5.0,3.6,1.4,0.2,Setosa
5.4,3.9,1.7,0.4,Setosa
4.6,3.4,1.4,0.3,Setosa
5.0,3.4,1.5,0.2,Setosa
4.4,2.9,1.4,0.2,Setosa
4.9,3.1,1.5,0.1,Setosa
7.0,3.2,4.7,1.4,Versicolor
6.4,3.2,4.5,1.5,Versicolor
6.9,3.1,4.9,1.5,Versicolor
5.5,2.3,4.0,1.3,Versicolor
6.5,2.8,4.6,1.5,Versicolor
5.7,2.8,4.5,1.3,Versicolor
6.3,3.3,4.7,1.6,Versicolor
4.9,2.4,3.3,1.0,Versicolor
6.6,2.9,4.6,1.3,Versicolor
5.2,2.7,3.9,1.4,Versicolor
6.3,3.3,6.0,2.5,Virginica
5.8,2.7,5.1,1.9,Virginica
7.1,3.0,5.9,2.1,Virginica
6.3,2.9,5.6,1.8,Virginica
6.5,3.0,5.8,2.2,Virginica
7.6,3.0,6.6,2.1,Virginica
4.9,2.5,4.5,1.7,Virginica
7.3,2.9,6.3,1.8,Virginica
6.7,2.5,5.8,1.8,Virginica
7.2,3.6,6.1,2.5,Virginica
6.5,3.2,5.1,2.0,Virginica
6.4,2.8,5.3,1.9,Virginica
6.8,3.0,5.5,2.1,Virginica
5.7,2.5,5.0,2.0,Virginica
5.8,2.8,5.1,2.4,Virginica
6.4,3.2,5.3,2.3,Virginica
6.5,3.0,5.5,1.8,Virginica
7.7,3.8,6.7,2.2,Virginica
7.7,2.6,6.9,2.3,Virginica
6.0,2.2,5.0,1.5,Virginica
6.9,3.2,5.7,2.3,Virginica
5.6,2.8,4.9,2.0,Virginica
7.7,2.8,6.7,2.0,Virginica
6.3,2.7,4.9,1.8,Virginica
6.7,3.3,5.7,2.1,Virginica
7.2,3.2,6.0,1.8,Virginica
6.2,2.8,4.8,1.8,Virginica
6.1,3.0,4.9,1.8,Virginica
6.4,2.8,5.6,2.1,Virginica
7.2,3.0,5.8,1.6,Virginica`
  },
  {
    id: "housing",
    name: "House Price Predictor",
    description: "A regression dataset containing physical property sizes, age, room counts, and historical sales prices. Ideal for testing linear models and tree regressors.",
    targetCol: "price",
    problemType: "regression",
    csvData: `square_feet,bedrooms,bathrooms,age_years,has_garage,price
1200,2,1,15,0,210000
1500,3,2,10,1,285000
1800,3,2,8,1,330000
2400,4,3,5,1,450000
900,1,1,25,0,145000
2100,3,2.5,12,1,360000
1650,3,2,20,1,265000
2800,4,3.5,2,1,540000
1100,2,1,30,0,165000
3200,5,4,4,1,620000
1350,3,1.5,18,0,230000
1950,3,2,6,1,355000
2200,4,2.5,14,1,385000
850,2,1,40,0,135000
2600,4,3,7,1,480000
1400,3,2,11,1,255000
1750,3,2,15,1,295000
2300,4,3,8,1,420000
3100,5,3.5,3,1,585000
1050,2,1,22,0,175000
2000,3,2.5,9,1,365000
1550,3,2,13,0,270000
2700,4,3,5,1,495000
1250,2,1.5,17,1,215000
2900,4,3.5,1,1,560000
950,1,1,28,0,150000
2250,4,3,10,1,410000
1600,3,2,12,1,280000
1850,3,2,7,1,340000
2500,4,3,4,1,475000
1150,2,1,24,0,180000
3400,5,4,2,1,650000
1300,3,1.5,16,0,225000
1900,3,2,8,1,350000
2150,4,2.5,11,1,390000
900,2,1,35,0,140000
2750,4,3,3,1,510000
1450,3,2,10,1,260000
1700,3,2,14,1,290000
2450,4,3,6,1,460000
3000,5,3.5,4,1,570000
1000,2,1,25,0,170000
2050,3,2.5,8,1,375000
1500,3,2,15,0,265000
2650,4,3,5,1,485000
1200,2,1.5,19,1,210000
2850,4,3.5,2,1,550000
800,1,1,32,0,125000
2200,4,3,9,1,405000
1550,3,2,11,1,275000`
  },
  {
    id: "heart_disease",
    name: "Heart Health Classifier",
    description: "A physical clinical factors dataset predicting whether a patient belongs to a 'High' or 'Low' cardiovascular risk group.",
    targetCol: "risk",
    problemType: "classification",
    csvData: `age,blood_pressure,cholesterol,max_heart_rate,exercise_angina,risk
52,125,212,168,0,Low
53,140,203,155,1,High
41,130,204,172,0,Low
56,120,236,178,0,Low
57,120,354,163,1,High
58,100,234,156,0,High
44,110,263,173,0,Low
52,128,205,184,0,Low
57,140,192,148,0,High
54,150,214,158,0,High
48,130,275,139,0,Low
49,130,266,171,0,Low
54,140,239,160,0,High
56,125,249,144,1,High
44,120,226,169,0,Low
52,135,223,161,0,Low
59,130,325,142,1,High
40,140,199,178,1,Low
51,120,295,157,0,Low
53,130,246,143,1,High
42,120,240,194,0,Low
43,135,233,181,0,Low
55,140,217,111,1,High
51,145,226,150,0,High
58,135,253,140,1,High
54,120,258,147,0,Low
47,112,204,143,0,Low
52,120,325,172,0,Low
55,130,262,155,0,Low
59,150,204,162,1,High
53,142,226,111,1,High
45,115,260,185,0,Low
56,130,256,142,1,High
50,110,254,159,0,Low
52,125,212,168,0,Low
53,140,203,155,1,High
41,130,204,172,0,Low
56,120,236,178,0,Low
57,120,354,163,1,High
58,100,234,156,0,High
44,110,263,173,0,Low
52,128,205,184,0,Low
57,140,192,148,0,High
54,150,214,158,0,High
48,130,275,139,0,Low
49,130,266,171,0,Low
54,140,239,160,0,High
56,125,249,144,1,High
44,120,226,169,0,Low
52,135,223,161,0,Low`
  }
];
