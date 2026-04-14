import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class LogPhasePicksCallback(pl.Callback):
    """
    Logs plots of waveforms with predicted and true P and S wave phase picks
    to MLflow or TensorBoard at the end of each validation epoch.
    """
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self.validation_batch = None
        
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Save only the first batch of the validation set to plot it consistently
        if batch_idx == 0:
            self.validation_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.validation_batch is None:
            return
            
        # Only plot and log on the main process to prevent deadlocks in DDP
        if getattr(trainer, "is_global_zero", False) is False:
            return
            
        # Get data
        x = self.validation_batch["input_values"].to(pl_module.device)
        y_true = self.validation_batch["labels"].to(pl_module.device)
        attention_mask = self.validation_batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(pl_module.device)
        
        # Predict
        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module(x, attention_mask)
            y_pred = outputs["probs"]
            
        # Plot
        num_to_plot = min(self.num_samples, x.shape[0])
        fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 3 * num_to_plot))
        if num_to_plot == 1:
            axes = [axes]
            
        for i in range(num_to_plot):
            ax = axes[i]
            
            # Assuming channel 0 is Z component, or we just plot the last channel if shape is [batch, channels, seq]
            if x.dim() == 3:
                waveform = x[i, -1].cpu().numpy() # Usually Z is last or first. Either way works.
            else:
                waveform = x[i].cpu().numpy()
                
            time_axis = range(len(waveform))
            
            # Plot waveform
            ax.plot(time_axis, waveform, color='black', alpha=0.4, label='Waveform')
            ax.set_ylabel("Amplitude")
            
            # Setup a twin axis for probabilities
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel("Probability")
            
            # True labels (assuming 1=P, 2=S from num_classes=3)
            p_true = y_true[i, 1].cpu().numpy()
            s_true = y_true[i, 2].cpu().numpy()
            ax2.plot(time_axis, p_true, color='blue', linestyle='--', alpha=0.7, label='True P')
            ax2.plot(time_axis, s_true, color='red', linestyle='--', alpha=0.7, label='True S')
            
            # Predicted labels
            p_pred = y_pred[i, 1].cpu().numpy()
            s_pred = y_pred[i, 2].cpu().numpy()
            ax2.plot(time_axis, p_pred, color='blue', label='Pred P')
            ax2.plot(time_axis, s_pred, color='red', label='Pred S')
            
            if i == 0:
                # Add legend to the first plot
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
            ax.set_title(f"Phase Picks - Validation Sample {i+1} (Epoch {trainer.current_epoch})")
            
        plt.tight_layout()
        
        # Log figure
        logger = trainer.logger
        if logger is not None:
            # Check if it's MLFlowLogger
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_figure"):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig, 
                    artifact_file=f"plots/val_picks_epoch_{trainer.current_epoch:03d}.png"
                )
            # Check if it's TensorBoardLogger
            elif hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
                logger.experiment.add_figure("val_picks", fig, global_step=trainer.current_epoch)
        
        plt.close(fig)
