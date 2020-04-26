from sklearn.linear_model import LinearRegression



x=[[4], [8], [12], [16], [18]]
y=[[4], [8], [10], [12], [15]]


model = LinearRegression()
model.fit(x,y)


print(model.predict([[7]]))
