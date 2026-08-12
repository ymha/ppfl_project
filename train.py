import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


def train_dp_sgd(peft_model, data_loader, target_delta, args):
    # Only optimize the LoRA A/B matrices -- everything else is frozen. This
    # plain AdamW instance is what make_private_with_epsilon below wraps into
    # dp_optimizer -- it is never stepped directly itself.
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    adamw_optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    privacy_engine = PrivacyEngine()
    # make_private_with_epsilon wraps (model, optimizer, data_loader) into
    # DP-SGD-aware versions:
    #   - dp_model: a GradSampleModule that hooks every trainable nn.Linear
    #     (here, only the LoRA layers) to compute per-sample gradients.
    #   - dp_optimizer: a DPOptimizer wrapping adamw_optimizer that clips each
    #     sample's gradient to max_grad_norm, sums them, adds Gaussian noise,
    #     then does a normal AdamW step.
    #   - dp_data_loader: a Poisson-sampling loader required for the privacy
    #     accountant's math to be valid.
    # The noise multiplier is solved for automatically so that, after `epochs`
    # passes over the data, the privacy spend equals target_epsilon.
    dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private_with_epsilon(
        module=peft_model,
        optimizer=adamw_optimizer,
        data_loader=data_loader,
        target_epsilon=args.target_epsilon,
        target_delta=target_delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )
    print(f"Using noise_multiplier={dp_optimizer.noise_multiplier:.4f} for target epsilon={args.target_epsilon}")

    device = torch.device("cuda")
    dp_model.train()
    step_count = 0
    # Splits each logical (DP-accounting) batch drawn from dp_data_loader into
    # physical chunks of at most args.batch_size before forward/backward.
    # dp_optimizer still only clips+noises+steps once per logical batch (it
    # tracks accumulation internally); only the peak GPU memory changes, not
    # the privacy accounting. The loop body below is unchanged from a plain
    # DP-SGD loop -- BatchMemoryManager is what makes physical batches "just
    # work" as if they were the full logical batch.
    with BatchMemoryManager(
        data_loader=dp_data_loader,
        max_physical_batch_size=args.batch_size,
        optimizer=dp_optimizer,
    ) as memory_safe_data_loader:
        for epoch in range(args.epochs):
            for step, batch in enumerate(memory_safe_data_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                dp_optimizer.zero_grad()
                # Forward/backward here trigger Opacus's hooks: per-sample grads
                # are captured for the LoRA layers, then clipped + noised inside
                # dp_optimizer.step() below (not a plain AdamW step).
                loss = dp_model(**batch).loss
                loss.backward()
                dp_optimizer.step()
                step_count += 1

                if step % args.log_every == 0:
                    # privacy_engine.accountant.history is only appended to on
                    # a *real* DP step (clip+noise actually applied) -- with
                    # BatchMemoryManager, most physical sub-batches just
                    # accumulate and don't trigger one, so history can still
                    # be empty this early (e.g. physical step 0, before the
                    # first logical/DP-accounting batch has finished
                    # accumulating). get_epsilon() divides by the recorded
                    # noise history and NaNs out if called before that.
                    if privacy_engine.accountant.history:
                        eps = privacy_engine.get_epsilon(delta=target_delta)
                        print(f"epoch {epoch} step {step}: loss={loss.item():.4f} eps={eps:.2f}")
                    else:
                        print(f"epoch {epoch} step {step}: loss={loss.item():.4f} eps=(no DP step yet)")

                if args.max_steps and step_count >= args.max_steps:
                    # Stopping early means less privacy budget was actually spent
                    # than target_epsilon called for -- get_epsilon() below still
                    # reports the true (smaller) spend for these step_count steps.
                    break
            if args.max_steps and step_count >= args.max_steps:
                break

    if privacy_engine.accountant.history:
        final_eps = privacy_engine.get_epsilon(delta=target_delta)
    else:
        # No full logical batch completed (e.g. --max-steps cut off training
        # before one accumulated) -- no DP guarantee was actually spent yet.
        final_eps = 0.0
    print(f"Final: epsilon={final_eps:.2f}, delta={target_delta:.2e}")
    return final_eps


def train_adamw(peft_model, data_loader, args):
    # Same LoRA update rule as train_dp_sgd(), just without Opacus: no
    # per-sample gradients, no clipping, no noise -- a normal AdamW step on
    # the batch gradient. This is the non-private utility upper bound for
    # comparison against the DP-SGD run.
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    adamw_optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    device = torch.device("cuda")
    peft_model.train()
    step_count = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            adamw_optimizer.zero_grad()
            loss = peft_model(**batch).loss
            loss.backward()
            adamw_optimizer.step()
            step_count += 1

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}")

            if args.max_steps and step_count >= args.max_steps:
                return
