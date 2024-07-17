
class RecommendationAlgorithm:
    def __init__(self):
        self.model = None
        self.age = 0
        self.gender = 0
        self.recommendation = ""
    
    def run_recommendation(self, age, gender):
        if age<10:
            self.age = 0
        elif 10<=age and age<20:
            self.age = 10
        elif 20<=age and age<30:
            self.age = 20
        elif 30<=age and age<40:
            self.age = 30
        elif 40<=age and age<50:
            self.age = 40
        elif 50<=age and age<60:
            self.age = 50
        else:
            self.age = 60

        self.gender = gender

        recommendation_map = {
            #(나이,성별): 추천메뉴 (아메리카노, 카페라떼 제외)
            (0, 0): "Caffe Latte",
            (10, 0): "Caffe Latte",
            (20, 0): "Grapefruit Honey Black Tea",
            (30, 0): "Dolce Cold Brew",
            (40, 0): "Dolce Latte",
            (50, 0): "Americano",
            (50, 0): "Americano",

            (0,0): "Caffe Latte",
            (10, 1): "java chip frappuccino",
            (20, 1): "Grapefruit Honey Black Tea",
            (30, 1): "Dolce Cold Brew",
            (40, 1): "Dolce Latte",
            (50, 1): "Decaf Americano",
            (50, 1): "Americano",
        }
        self.recommendation = recommendation_map[(self.age, self.gender)]
    
    def get_recommendation(self):
        return self.recommendation
