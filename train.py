from torch.nn import L1Loss
from ignite.contrib.metrics.regression.r2_score import R2Score
import tqdm

class Loss:
    def __init__(self, model, dataset, train=False, optimizer=None):
        self.model = model
        self.dataset = dataset
        self.train = train
        self.optimizer = optimizer

    def model_loss(self):
        # first calculated for the batches and at the end get the average
        performance = L1Loss()
        score_metric = R2Score()

        avg_loss = 0
        avg_score = 0
        count = 0
        pred = []
        out = []
        name = []

        for input, output, image_name in iter(tqdm.tqdm(self.dataset)):
            # get predictions of the model for training set
            input, output = input.cuda(), output.cuda()
            output = output.reshape(output.shape[0], 1).float()
            predictions = self.model.forward(input)
            for i in predictions:
                pred.append(i.item())
            for i in output:
                out.append(i.item())
            for i in image_name:
                name.append(i)

            # calculate loss of the model
            loss = performance(predictions, output)

            # compute the R2 score
            score_metric.update([predictions, output])
            score = score_metric.compute()

            if self.train:
                # clear the errors
                self.optimizer.zero_grad()

                # compute the gradients for optimizer
                loss.backward()

                # use optimizer in order to update parameters
                # of the model based on gradients
                self.optimizer.step()

            # store the loss and update values
            avg_loss += loss.item()
            avg_score += score
            count += 1

        return avg_loss / count, avg_score / count, predictions, output, pred, out, name
