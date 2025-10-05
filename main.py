# model class
class Model:
    # initializing the default variables
    def __init__(self, feature_count, learning_rate=0.01):
        self.weights = [0.005 for _ in range(feature_count)]  # small positive weights
        self.bias = 0.0043
        self.learning_rate = learning_rate

    def predict(self, feature_row):
        return sum(w * x for w, x in zip(self.weights, feature_row)) + self.bias
    # Step 1: calculate y
    def calculate_y(self, feature_row):
        predicted_price = sum(w * x for w, x in zip(self.weights, feature_row)) + self.bias
        return predicted_price

    # Step 2: compute error (difference)
    def compute_error(self, predicted_price, Y):
        return predicted_price - Y  # simple difference, not squared

    # Step 3: compute gradients for weights and bias
    def compute_loss_function(self, E, feature_row):
        weight_gradients = [2 * E * x for x in feature_row]
        bias_gradient = 2 * E
        return weight_gradients, bias_gradient

    # Step 4: update all weights and bias
    def update_weights_and_bias(self, weight_gradients, bias_gradient):
        self.weights = [w - self.learning_rate * gw for w, gw in zip(self.weights, weight_gradients)]
        self.bias -= self.learning_rate * bias_gradient

    # ==== TRAINING METHOD ====
    def train(self, features, prices, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for feature_row, Y in zip(features, prices):
                y_pred = self.calculate_y(feature_row)
                E = self.compute_error(y_pred, Y)
                weight_grad, bias_grad = self.compute_loss_function(E, feature_row)
                self.update_weights_and_bias(weight_grad, bias_grad)
                total_loss += E ** 2  # sum squared error for reporting

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("\nTraining finished!")
        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias:.4f}")


def main() -> None:
    # Example normalized features and prices
    features = [
        [0.85, 0.75, 0.10, 0.20, 0.30],
        [0.60, 0.50, 0.40, 0.35, 0.50],
        [0.90, 0.80, 0.05, 0.15, 0.25],
        [0.30, 0.40, 0.60, 0.50, 0.70],
    ]
    prices = [0.92, 0.75, 0.95, 0.55]

    model = Model(feature_count=5, learning_rate=0.01)
    model.train(features, prices, epochs=1000)

    new_house=[0.70, 0.60, 0.20, 0.30, 0.40]
    predicted_price=model.predict(new_house)
    print(f"Predicted Price of new house: {predicted_price:.2f}")

if __name__ == '__main__':
    main()
