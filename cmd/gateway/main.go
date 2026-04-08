package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lunargate-ai/gateway/internal/api"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/remotecontrol"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var version = "dev"

func main() {
	configPath := flag.String("config", "config.yaml", "Path to config file")
	logLevel := flag.String("log-level", "", "Override log level (debug, info, warn, error)")
	flag.Parse()

	// --- Load Config ---
	cfgManager, err := config.NewManager(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	cfg := cfgManager.Get()
	if *logLevel != "" {
		cfg.Logging.Level = *logLevel
	}

	// --- Setup Logging ---
	effectiveLogLevel := setupLogging(cfg.Logging)
	printStartupBanner(version, *configPath, effectiveLogLevel, cfg.Server.Address())

	log.Info().
		Str("version", version).
		Str("config", *configPath).
		Str("log_level", effectiveLogLevel).
		Msg("starting LunarGate gateway")

	// --- Initialize Components ---
	registry := providers.NewRegistry(cfg.Providers)
	if len(registry.List()) == 0 {
		log.Fatal().Msg("no providers configured - at least one provider is required")
	}

	routingEngine := routing.NewEngine(cfg.Routing)
	retrier := resilience.NewRetrier(cfg.Retry)
	cbManager := resilience.NewCircuitBreakerManager()
	fallbackExec := resilience.NewFallbackExecutor(retrier, cbManager)
	cache := middleware.NewCache(cfg.Cache)
	rateLimiter := middleware.NewRateLimiter(cfg.RateLimit)
	streamer := streaming.NewHandler()
	metrics := observability.NewMetrics()
	healthChecker := health.NewChecker(version)
	collectorClient := observability.NewCollectorClient(cfg.DataSharing, version)
	selector := modelselect.NewEngine(cfg.ModelSelect)
	store := modelstore.NewStore(registry, cfg.Providers)
	localBaseURL := "http://" + localLoopbackAddress(cfg.Server)
	remoteControlClient := remotecontrol.NewClient(
		cfg.DataSharing,
		version,
		localBaseURL,
		routingEngine.RouteNames,
		func(ctx context.Context) []string {
			var ids []string
			if store != nil {
				for _, item := range store.AllModels(ctx) {
					if item.ID != "" {
						ids = append(ids, item.ID)
					}
				}
			}
			return ids
		},
	)
	remoteControlEnabled := cfg.DataSharing.RemoteControl && remoteControlClient != nil
	remoteControlInstanceID := ""
	if remoteControlClient != nil {
		remoteControlInstanceID = remoteControlClient.InstanceID()
	}

	log.Info().
		Bool("data_sharing_enabled", cfg.DataSharing.Enabled).
		Bool("remote_control_enabled", remoteControlEnabled).
		Str("gateway_id", cfg.DataSharing.GatewayID).
		Str("instance_id", remoteControlInstanceID).
		Str("backend_url", cfg.DataSharing.BackendURL).
		Msg("gateway data sharing and remote control status")

	// --- Setup Hot-Reload ---
	cfgManager.OnChange(func(newCfg *config.Config) {
		routingEngine.UpdateConfig(newCfg.Routing)
		rateLimiter.UpdateConfig(newCfg.RateLimit)
		selector.UpdateConfig(newCfg.ModelSelect)
		store.UpdateProvidersConfig(newCfg.Providers)
		if remoteControlClient != nil {
			remoteControlClient.RefreshHello()
		}
		log.Info().Msg("hot-reload: routing, rate limit, model selection, and model store config updated")
	})
	cfgManager.WatchChanges()

	// --- Create API Handler & Router ---
	handler := api.NewHandler(registry, routingEngine, fallbackExec, cache, streamer, metrics, collectorClient, selector, store)
	router := api.NewRouter(handler, rateLimiter, healthChecker)

	// --- Start HTTP Server ---
	srv := &http.Server{
		Addr:              cfg.Server.Address(),
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.Server.ReadTimeout,
		WriteTimeout:      cfg.Server.WriteTimeout,
		IdleTimeout:       cfg.Server.IdleTimeout,
		MaxHeaderBytes:    1 << 20,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Info().
			Str("address", cfg.Server.Address()).
			Strs("providers", registry.List()).
			Msg("gateway listening")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("server failed")
		}
	}()

	remoteControlCtx, remoteControlCancel := context.WithCancel(context.Background())
	defer remoteControlCancel()
	if remoteControlClient != nil {
		remoteControlClient.Start(remoteControlCtx)
	}

	<-done
	log.Info().Msg("shutting down gateway...")

	healthChecker.SetReady(false)
	remoteControlCancel()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server shutdown error")
	}
	if collectorClient != nil {
		collectorClient.Stop()
	}
	if cache != nil {
		cache.Stop()
	}

	log.Info().Msg("gateway stopped")
}

func setupLogging(cfg config.LoggingConfig) string {
	level, err := zerolog.ParseLevel(cfg.Level)
	if err != nil {
		level = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(level)

	if cfg.Format == "console" {
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339})
	}

	zerolog.TimeFieldFormat = time.RFC3339Nano

	return level.String()
}

func printStartupBanner(version, configPath, logLevel, address string) {
	fmt.Fprintf(os.Stderr, `
 _                           ____       _
| |   _   _ _ __   __ _ _ _ / ___| __ _| |_ ___
| |  | | | | '_ \ / _`+"`"+` | '_| |  _ / _`+"`"+` | __/ _ \
| |__| |_| | | | | (_| | | | |_| | (_| | ||  __/
|_____\__,_|_| |_|\__,_|_|  \____|\__,_|\__\___|

version: %s
config:  %s
log:     %s
listen:  %s

`, version, configPath, logLevel, address)
}

func localLoopbackAddress(cfg config.ServerConfig) string {
	host := cfg.Host
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return fmt.Sprintf("%s:%d", host, cfg.Port)
}
